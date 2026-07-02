#include <type_traits>
#include "../poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_local.hpp"
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
struct poro_hyperelasticity_poro_form_1_u_affine_reference_data {
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
    static const scalar_t *hex27_shape_1d() {
        static const scalar_t data[12] = {scalar_t(0.80134602936993082), scalar_t(0.25844425285419081), scalar_t(-0.059790282224121687), scalar_t(0.22778407679095203), scalar_t(0.88441289000295209), scalar_t(-0.11219696679390417), scalar_t(-0.11219696679390401), scalar_t(0.88441289000295198), scalar_t(0.22778407679095214), scalar_t(-0.05979028222412186), scalar_t(0.25844425285419081), scalar_t(0.80134602936993082)};
        return data;
    }
    static const scalar_t *hex27_grad_1d() {
        static const scalar_t data[12] = {scalar_t(-2.7222726231881049), scalar_t(3.4445452463762103), scalar_t(-0.72227262318810515), scalar_t(-1.6799620871697125), scalar_t(1.359924174339425), scalar_t(0.32003791283028749), scalar_t(-0.32003791283028749), scalar_t(-1.359924174339425), scalar_t(1.6799620871697125), scalar_t(0.72227262318810492), scalar_t(-3.4445452463762098), scalar_t(2.7222726231881049)};
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
struct poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data {
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
    static const scalar_t *hex27_shape_1d() {
        static const scalar_t data[12] = {scalar_t(0.80134602936993082), scalar_t(0.25844425285419081), scalar_t(-0.059790282224121687), scalar_t(0.22778407679095203), scalar_t(0.88441289000295209), scalar_t(-0.11219696679390417), scalar_t(-0.11219696679390401), scalar_t(0.88441289000295198), scalar_t(0.22778407679095214), scalar_t(-0.05979028222412186), scalar_t(0.25844425285419081), scalar_t(0.80134602936993082)};
        return data;
    }
    static const scalar_t *hex27_grad_1d() {
        static const scalar_t data[12] = {scalar_t(-2.7222726231881049), scalar_t(3.4445452463762103), scalar_t(-0.72227262318810515), scalar_t(-1.6799620871697125), scalar_t(1.359924174339425), scalar_t(0.32003791283028749), scalar_t(-0.32003791283028749), scalar_t(-1.359924174339425), scalar_t(1.6799620871697125), scalar_t(0.72227262318810492), scalar_t(-3.4445452463762098), scalar_t(2.7222726231881049)};
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

static const KernelDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa",
    "HEX27",
    3,
    64,
    27,
    16,
    4,
    10,
    20,
    1,
    0,
    0,
    0,
    0,
    0,
    22,
    5,
    38,
    0,
    0,
    1,
    19,
    10,
    40,
    4,
    1,
    89,
    0,
    89,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa",
    "HEX27",
    3,
    64,
    27,
    16,
    4,
    6,
    20,
    1,
    0,
    0,
    0,
    0,
    0,
    18,
    7,
    34,
    0,
    0,
    3,
    14,
    10,
    40,
    4,
    0,
    0,
    0,
    89,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_mixed_impl(
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
        const scalar_t alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u_data[3],
        const scalar_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_1_u_affine_reference_data<scalar_t>::hex27_shape_1d(), sfem::codegen::poro_hyperelasticity_poro_form_1_u_affine_reference_data<scalar_t>::hex8_shape_1d()};
    const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_1_u_affine_reference_data<scalar_t>::hex27_grad_1d(), sfem::codegen::poro_hyperelasticity_poro_form_1_u_affine_reference_data<scalar_t>::hex8_grad_1d()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
            ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
            ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
            ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
            ev[4 * VECTOR_SIZE + lane] = elements[4][evbegin + lane];
            ev[5 * VECTOR_SIZE + lane] = elements[5][evbegin + lane];
            ev[6 * VECTOR_SIZE + lane] = elements[6][evbegin + lane];
            ev[7 * VECTOR_SIZE + lane] = elements[7][evbegin + lane];
            ev[8 * VECTOR_SIZE + lane] = elements[8][evbegin + lane];
            ev[9 * VECTOR_SIZE + lane] = elements[9][evbegin + lane];
            ev[10 * VECTOR_SIZE + lane] = elements[10][evbegin + lane];
            ev[11 * VECTOR_SIZE + lane] = elements[11][evbegin + lane];
            ev[12 * VECTOR_SIZE + lane] = elements[12][evbegin + lane];
            ev[13 * VECTOR_SIZE + lane] = elements[13][evbegin + lane];
            ev[14 * VECTOR_SIZE + lane] = elements[14][evbegin + lane];
            ev[15 * VECTOR_SIZE + lane] = elements[15][evbegin + lane];
            ev[16 * VECTOR_SIZE + lane] = elements[16][evbegin + lane];
            ev[17 * VECTOR_SIZE + lane] = elements[17][evbegin + lane];
            ev[18 * VECTOR_SIZE + lane] = elements[18][evbegin + lane];
            ev[19 * VECTOR_SIZE + lane] = elements[19][evbegin + lane];
            ev[20 * VECTOR_SIZE + lane] = elements[20][evbegin + lane];
            ev[21 * VECTOR_SIZE + lane] = elements[21][evbegin + lane];
            ev[22 * VECTOR_SIZE + lane] = elements[22][evbegin + lane];
            ev[23 * VECTOR_SIZE + lane] = elements[23][evbegin + lane];
            ev[24 * VECTOR_SIZE + lane] = elements[24][evbegin + lane];
            ev[25 * VECTOR_SIZE + lane] = elements[25][evbegin + lane];
            ev[26 * VECTOR_SIZE + lane] = elements[26][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_current[0][lane] = u_data[0][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[1][lane] = u_data[0][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[2][lane] = u_data[0][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[3][lane] = u_data[0][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[4][lane] = u_data[0][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[5][lane] = u_data[0][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[6][lane] = u_data[0][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[7][lane] = u_data[0][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[8][lane] = u_data[0][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[9][lane] = u_data[0][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[10][lane] = u_data[0][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[11][lane] = u_data[0][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[12][lane] = u_data[0][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[13][lane] = u_data[0][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[14][lane] = u_data[0][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[15][lane] = u_data[0][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[16][lane] = u_data[0][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[17][lane] = u_data[0][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[18][lane] = u_data[0][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[19][lane] = u_data[0][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[20][lane] = u_data[0][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[21][lane] = u_data[0][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[22][lane] = u_data[0][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[23][lane] = u_data[0][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[24][lane] = u_data[0][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[25][lane] = u_data[0][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[26][lane] = u_data[0][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[27][lane] = u_data[1][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[28][lane] = u_data[1][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[29][lane] = u_data[1][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[30][lane] = u_data[1][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[31][lane] = u_data[1][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[32][lane] = u_data[1][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[33][lane] = u_data[1][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[34][lane] = u_data[1][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[35][lane] = u_data[1][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[36][lane] = u_data[1][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[37][lane] = u_data[1][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[38][lane] = u_data[1][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[39][lane] = u_data[1][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[40][lane] = u_data[1][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[41][lane] = u_data[1][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[42][lane] = u_data[1][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[43][lane] = u_data[1][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[44][lane] = u_data[1][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[45][lane] = u_data[1][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[46][lane] = u_data[1][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[47][lane] = u_data[1][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[48][lane] = u_data[1][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[49][lane] = u_data[1][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[50][lane] = u_data[1][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[51][lane] = u_data[1][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[52][lane] = u_data[1][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[53][lane] = u_data[1][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[54][lane] = u_data[2][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[55][lane] = u_data[2][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[56][lane] = u_data[2][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[57][lane] = u_data[2][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[58][lane] = u_data[2][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[59][lane] = u_data[2][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[60][lane] = u_data[2][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[61][lane] = u_data[2][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[62][lane] = u_data[2][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[63][lane] = u_data[2][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[64][lane] = u_data[2][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[65][lane] = u_data[2][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[66][lane] = u_data[2][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[67][lane] = u_data[2][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[68][lane] = u_data[2][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[69][lane] = u_data[2][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[70][lane] = u_data[2][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[71][lane] = u_data[2][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[72][lane] = u_data[2][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[73][lane] = u_data[2][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[74][lane] = u_data[2][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[75][lane] = u_data[2][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[76][lane] = u_data[2][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[77][lane] = u_data[2][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[78][lane] = u_data[2][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[79][lane] = u_data[2][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[80][lane] = u_data[2][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[81][lane] = p_data[ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[82][lane] = p_data[ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[83][lane] = p_data[ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[84][lane] = p_data[ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[85][lane] = p_data[ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[86][lane] = p_data[ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[87][lane] = p_data[ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[88][lane] = p_data[ev[7 * VECTOR_SIZE + lane] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
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
        const scalar_t *const block_current_streams[N_FIELD_STREAMS] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63], block_current[64], block_current[65], block_current[66], block_current[67], block_current[68], block_current[69], block_current[70], block_current[71], block_current[72], block_current[73], block_current[74], block_current[75], block_current[76], block_current[77], block_current[78], block_current[79], block_current[80], block_current[81], block_current[82], block_current[83], block_current[84], block_current[85], block_current[86], block_current[87], block_current[88]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88]};

        poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_residual_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::poro_hyperelasticity_poro_form_1_u_affine_reference_data<scalar_t>::q_weight_1d(), block_current_streams, alpha, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[88][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa(
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
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_soa_float(
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
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u_data[3],
        const scalar_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex27_shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex27_grad_1d();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
            ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
            ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
            ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
            ev[4 * VECTOR_SIZE + lane] = elements[4][evbegin + lane];
            ev[5 * VECTOR_SIZE + lane] = elements[5][evbegin + lane];
            ev[6 * VECTOR_SIZE + lane] = elements[6][evbegin + lane];
            ev[7 * VECTOR_SIZE + lane] = elements[7][evbegin + lane];
            ev[8 * VECTOR_SIZE + lane] = elements[8][evbegin + lane];
            ev[9 * VECTOR_SIZE + lane] = elements[9][evbegin + lane];
            ev[10 * VECTOR_SIZE + lane] = elements[10][evbegin + lane];
            ev[11 * VECTOR_SIZE + lane] = elements[11][evbegin + lane];
            ev[12 * VECTOR_SIZE + lane] = elements[12][evbegin + lane];
            ev[13 * VECTOR_SIZE + lane] = elements[13][evbegin + lane];
            ev[14 * VECTOR_SIZE + lane] = elements[14][evbegin + lane];
            ev[15 * VECTOR_SIZE + lane] = elements[15][evbegin + lane];
            ev[16 * VECTOR_SIZE + lane] = elements[16][evbegin + lane];
            ev[17 * VECTOR_SIZE + lane] = elements[17][evbegin + lane];
            ev[18 * VECTOR_SIZE + lane] = elements[18][evbegin + lane];
            ev[19 * VECTOR_SIZE + lane] = elements[19][evbegin + lane];
            ev[20 * VECTOR_SIZE + lane] = elements[20][evbegin + lane];
            ev[21 * VECTOR_SIZE + lane] = elements[21][evbegin + lane];
            ev[22 * VECTOR_SIZE + lane] = elements[22][evbegin + lane];
            ev[23 * VECTOR_SIZE + lane] = elements[23][evbegin + lane];
            ev[24 * VECTOR_SIZE + lane] = elements[24][evbegin + lane];
            ev[25 * VECTOR_SIZE + lane] = elements[25][evbegin + lane];
            ev[26 * VECTOR_SIZE + lane] = elements[26][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[0 * VECTOR_SIZE + lane]];
            block_coordinates[1][lane] = points[1][ev[0 * VECTOR_SIZE + lane]];
            block_coordinates[2][lane] = points[2][ev[0 * VECTOR_SIZE + lane]];
            block_coordinates[3][lane] = points[0][ev[1 * VECTOR_SIZE + lane]];
            block_coordinates[4][lane] = points[1][ev[1 * VECTOR_SIZE + lane]];
            block_coordinates[5][lane] = points[2][ev[1 * VECTOR_SIZE + lane]];
            block_coordinates[6][lane] = points[0][ev[2 * VECTOR_SIZE + lane]];
            block_coordinates[7][lane] = points[1][ev[2 * VECTOR_SIZE + lane]];
            block_coordinates[8][lane] = points[2][ev[2 * VECTOR_SIZE + lane]];
            block_coordinates[9][lane] = points[0][ev[3 * VECTOR_SIZE + lane]];
            block_coordinates[10][lane] = points[1][ev[3 * VECTOR_SIZE + lane]];
            block_coordinates[11][lane] = points[2][ev[3 * VECTOR_SIZE + lane]];
            block_coordinates[12][lane] = points[0][ev[4 * VECTOR_SIZE + lane]];
            block_coordinates[13][lane] = points[1][ev[4 * VECTOR_SIZE + lane]];
            block_coordinates[14][lane] = points[2][ev[4 * VECTOR_SIZE + lane]];
            block_coordinates[15][lane] = points[0][ev[5 * VECTOR_SIZE + lane]];
            block_coordinates[16][lane] = points[1][ev[5 * VECTOR_SIZE + lane]];
            block_coordinates[17][lane] = points[2][ev[5 * VECTOR_SIZE + lane]];
            block_coordinates[18][lane] = points[0][ev[6 * VECTOR_SIZE + lane]];
            block_coordinates[19][lane] = points[1][ev[6 * VECTOR_SIZE + lane]];
            block_coordinates[20][lane] = points[2][ev[6 * VECTOR_SIZE + lane]];
            block_coordinates[21][lane] = points[0][ev[7 * VECTOR_SIZE + lane]];
            block_coordinates[22][lane] = points[1][ev[7 * VECTOR_SIZE + lane]];
            block_coordinates[23][lane] = points[2][ev[7 * VECTOR_SIZE + lane]];
            block_coordinates[24][lane] = points[0][ev[8 * VECTOR_SIZE + lane]];
            block_coordinates[25][lane] = points[1][ev[8 * VECTOR_SIZE + lane]];
            block_coordinates[26][lane] = points[2][ev[8 * VECTOR_SIZE + lane]];
            block_coordinates[27][lane] = points[0][ev[9 * VECTOR_SIZE + lane]];
            block_coordinates[28][lane] = points[1][ev[9 * VECTOR_SIZE + lane]];
            block_coordinates[29][lane] = points[2][ev[9 * VECTOR_SIZE + lane]];
            block_coordinates[30][lane] = points[0][ev[10 * VECTOR_SIZE + lane]];
            block_coordinates[31][lane] = points[1][ev[10 * VECTOR_SIZE + lane]];
            block_coordinates[32][lane] = points[2][ev[10 * VECTOR_SIZE + lane]];
            block_coordinates[33][lane] = points[0][ev[11 * VECTOR_SIZE + lane]];
            block_coordinates[34][lane] = points[1][ev[11 * VECTOR_SIZE + lane]];
            block_coordinates[35][lane] = points[2][ev[11 * VECTOR_SIZE + lane]];
            block_coordinates[36][lane] = points[0][ev[12 * VECTOR_SIZE + lane]];
            block_coordinates[37][lane] = points[1][ev[12 * VECTOR_SIZE + lane]];
            block_coordinates[38][lane] = points[2][ev[12 * VECTOR_SIZE + lane]];
            block_coordinates[39][lane] = points[0][ev[13 * VECTOR_SIZE + lane]];
            block_coordinates[40][lane] = points[1][ev[13 * VECTOR_SIZE + lane]];
            block_coordinates[41][lane] = points[2][ev[13 * VECTOR_SIZE + lane]];
            block_coordinates[42][lane] = points[0][ev[14 * VECTOR_SIZE + lane]];
            block_coordinates[43][lane] = points[1][ev[14 * VECTOR_SIZE + lane]];
            block_coordinates[44][lane] = points[2][ev[14 * VECTOR_SIZE + lane]];
            block_coordinates[45][lane] = points[0][ev[15 * VECTOR_SIZE + lane]];
            block_coordinates[46][lane] = points[1][ev[15 * VECTOR_SIZE + lane]];
            block_coordinates[47][lane] = points[2][ev[15 * VECTOR_SIZE + lane]];
            block_coordinates[48][lane] = points[0][ev[16 * VECTOR_SIZE + lane]];
            block_coordinates[49][lane] = points[1][ev[16 * VECTOR_SIZE + lane]];
            block_coordinates[50][lane] = points[2][ev[16 * VECTOR_SIZE + lane]];
            block_coordinates[51][lane] = points[0][ev[17 * VECTOR_SIZE + lane]];
            block_coordinates[52][lane] = points[1][ev[17 * VECTOR_SIZE + lane]];
            block_coordinates[53][lane] = points[2][ev[17 * VECTOR_SIZE + lane]];
            block_coordinates[54][lane] = points[0][ev[18 * VECTOR_SIZE + lane]];
            block_coordinates[55][lane] = points[1][ev[18 * VECTOR_SIZE + lane]];
            block_coordinates[56][lane] = points[2][ev[18 * VECTOR_SIZE + lane]];
            block_coordinates[57][lane] = points[0][ev[19 * VECTOR_SIZE + lane]];
            block_coordinates[58][lane] = points[1][ev[19 * VECTOR_SIZE + lane]];
            block_coordinates[59][lane] = points[2][ev[19 * VECTOR_SIZE + lane]];
            block_coordinates[60][lane] = points[0][ev[20 * VECTOR_SIZE + lane]];
            block_coordinates[61][lane] = points[1][ev[20 * VECTOR_SIZE + lane]];
            block_coordinates[62][lane] = points[2][ev[20 * VECTOR_SIZE + lane]];
            block_coordinates[63][lane] = points[0][ev[21 * VECTOR_SIZE + lane]];
            block_coordinates[64][lane] = points[1][ev[21 * VECTOR_SIZE + lane]];
            block_coordinates[65][lane] = points[2][ev[21 * VECTOR_SIZE + lane]];
            block_coordinates[66][lane] = points[0][ev[22 * VECTOR_SIZE + lane]];
            block_coordinates[67][lane] = points[1][ev[22 * VECTOR_SIZE + lane]];
            block_coordinates[68][lane] = points[2][ev[22 * VECTOR_SIZE + lane]];
            block_coordinates[69][lane] = points[0][ev[23 * VECTOR_SIZE + lane]];
            block_coordinates[70][lane] = points[1][ev[23 * VECTOR_SIZE + lane]];
            block_coordinates[71][lane] = points[2][ev[23 * VECTOR_SIZE + lane]];
            block_coordinates[72][lane] = points[0][ev[24 * VECTOR_SIZE + lane]];
            block_coordinates[73][lane] = points[1][ev[24 * VECTOR_SIZE + lane]];
            block_coordinates[74][lane] = points[2][ev[24 * VECTOR_SIZE + lane]];
            block_coordinates[75][lane] = points[0][ev[25 * VECTOR_SIZE + lane]];
            block_coordinates[76][lane] = points[1][ev[25 * VECTOR_SIZE + lane]];
            block_coordinates[77][lane] = points[2][ev[25 * VECTOR_SIZE + lane]];
            block_coordinates[78][lane] = points[0][ev[26 * VECTOR_SIZE + lane]];
            block_coordinates[79][lane] = points[1][ev[26 * VECTOR_SIZE + lane]];
            block_coordinates[80][lane] = points[2][ev[26 * VECTOR_SIZE + lane]];
            block_current[0][lane] = u_data[0][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[1][lane] = u_data[0][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[2][lane] = u_data[0][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[3][lane] = u_data[0][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[4][lane] = u_data[0][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[5][lane] = u_data[0][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[6][lane] = u_data[0][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[7][lane] = u_data[0][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[8][lane] = u_data[0][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[9][lane] = u_data[0][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[10][lane] = u_data[0][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[11][lane] = u_data[0][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[12][lane] = u_data[0][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[13][lane] = u_data[0][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[14][lane] = u_data[0][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[15][lane] = u_data[0][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[16][lane] = u_data[0][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[17][lane] = u_data[0][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[18][lane] = u_data[0][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[19][lane] = u_data[0][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[20][lane] = u_data[0][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[21][lane] = u_data[0][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[22][lane] = u_data[0][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[23][lane] = u_data[0][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[24][lane] = u_data[0][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[25][lane] = u_data[0][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[26][lane] = u_data[0][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[27][lane] = u_data[1][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[28][lane] = u_data[1][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[29][lane] = u_data[1][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[30][lane] = u_data[1][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[31][lane] = u_data[1][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[32][lane] = u_data[1][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[33][lane] = u_data[1][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[34][lane] = u_data[1][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[35][lane] = u_data[1][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[36][lane] = u_data[1][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[37][lane] = u_data[1][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[38][lane] = u_data[1][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[39][lane] = u_data[1][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[40][lane] = u_data[1][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[41][lane] = u_data[1][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[42][lane] = u_data[1][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[43][lane] = u_data[1][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[44][lane] = u_data[1][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[45][lane] = u_data[1][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[46][lane] = u_data[1][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[47][lane] = u_data[1][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[48][lane] = u_data[1][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[49][lane] = u_data[1][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[50][lane] = u_data[1][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[51][lane] = u_data[1][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[52][lane] = u_data[1][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[53][lane] = u_data[1][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[54][lane] = u_data[2][ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[55][lane] = u_data[2][ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[56][lane] = u_data[2][ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[57][lane] = u_data[2][ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[58][lane] = u_data[2][ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[59][lane] = u_data[2][ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[60][lane] = u_data[2][ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[61][lane] = u_data[2][ev[7 * VECTOR_SIZE + lane] * current_stride];
            block_current[62][lane] = u_data[2][ev[8 * VECTOR_SIZE + lane] * current_stride];
            block_current[63][lane] = u_data[2][ev[9 * VECTOR_SIZE + lane] * current_stride];
            block_current[64][lane] = u_data[2][ev[10 * VECTOR_SIZE + lane] * current_stride];
            block_current[65][lane] = u_data[2][ev[11 * VECTOR_SIZE + lane] * current_stride];
            block_current[66][lane] = u_data[2][ev[12 * VECTOR_SIZE + lane] * current_stride];
            block_current[67][lane] = u_data[2][ev[13 * VECTOR_SIZE + lane] * current_stride];
            block_current[68][lane] = u_data[2][ev[14 * VECTOR_SIZE + lane] * current_stride];
            block_current[69][lane] = u_data[2][ev[15 * VECTOR_SIZE + lane] * current_stride];
            block_current[70][lane] = u_data[2][ev[16 * VECTOR_SIZE + lane] * current_stride];
            block_current[71][lane] = u_data[2][ev[17 * VECTOR_SIZE + lane] * current_stride];
            block_current[72][lane] = u_data[2][ev[18 * VECTOR_SIZE + lane] * current_stride];
            block_current[73][lane] = u_data[2][ev[19 * VECTOR_SIZE + lane] * current_stride];
            block_current[74][lane] = u_data[2][ev[20 * VECTOR_SIZE + lane] * current_stride];
            block_current[75][lane] = u_data[2][ev[21 * VECTOR_SIZE + lane] * current_stride];
            block_current[76][lane] = u_data[2][ev[22 * VECTOR_SIZE + lane] * current_stride];
            block_current[77][lane] = u_data[2][ev[23 * VECTOR_SIZE + lane] * current_stride];
            block_current[78][lane] = u_data[2][ev[24 * VECTOR_SIZE + lane] * current_stride];
            block_current[79][lane] = u_data[2][ev[25 * VECTOR_SIZE + lane] * current_stride];
            block_current[80][lane] = u_data[2][ev[26 * VECTOR_SIZE + lane] * current_stride];
            block_current[81][lane] = p_data[ev[0 * VECTOR_SIZE + lane] * current_stride];
            block_current[82][lane] = p_data[ev[1 * VECTOR_SIZE + lane] * current_stride];
            block_current[83][lane] = p_data[ev[2 * VECTOR_SIZE + lane] * current_stride];
            block_current[84][lane] = p_data[ev[3 * VECTOR_SIZE + lane] * current_stride];
            block_current[85][lane] = p_data[ev[4 * VECTOR_SIZE + lane] * current_stride];
            block_current[86][lane] = p_data[ev[5 * VECTOR_SIZE + lane] * current_stride];
            block_current[87][lane] = p_data[ev[6 * VECTOR_SIZE + lane] * current_stride];
            block_current[88][lane] = p_data[ev[7 * VECTOR_SIZE + lane] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
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

        const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex8_shape_1d()};
        const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex27_grad_1d(), sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::hex8_grad_1d()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_current_streams[N_FIELD_STREAMS] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63], block_current[64], block_current[65], block_current[66], block_current[67], block_current[68], block_current[69], block_current[70], block_current[71], block_current[72], block_current[73], block_current[74], block_current[75], block_current[76], block_current[77], block_current[78], block_current[79], block_current[80], block_current[81], block_current[82], block_current[83], block_current[84], block_current[85], block_current[86], block_current[87], block_current[88]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88]};

        poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_residual_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::poro_hyperelasticity_poro_form_1_u_isoparametric_reference_data<scalar_t>::q_weight_1d(), block_current_streams, alpha, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[8 * VECTOR_SIZE + scatter] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[9 * VECTOR_SIZE + scatter] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[10 * VECTOR_SIZE + scatter] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[11 * VECTOR_SIZE + scatter] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[12 * VECTOR_SIZE + scatter] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[13 * VECTOR_SIZE + scatter] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[14 * VECTOR_SIZE + scatter] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[15 * VECTOR_SIZE + scatter] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[16 * VECTOR_SIZE + scatter] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[17 * VECTOR_SIZE + scatter] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[18 * VECTOR_SIZE + scatter] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[19 * VECTOR_SIZE + scatter] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[20 * VECTOR_SIZE + scatter] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[21 * VECTOR_SIZE + scatter] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[22 * VECTOR_SIZE + scatter] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[23 * VECTOR_SIZE + scatter] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[24 * VECTOR_SIZE + scatter] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[25 * VECTOR_SIZE + scatter] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[26 * VECTOR_SIZE + scatter] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[4 * VECTOR_SIZE + scatter] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[5 * VECTOR_SIZE + scatter] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[6 * VECTOR_SIZE + scatter] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[7 * VECTOR_SIZE + scatter] * out_stride] += block_output[88][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, alpha, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, alpha, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
