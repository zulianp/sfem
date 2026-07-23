#include <type_traits>
#include "../navier_stokes_form_2_u_p_d3_tensor_product_mixed_local.hpp"
#include "../../../kernel_math.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"

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
struct navier_stokes_form_2_u_p_affine_reference_data {
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
struct navier_stokes_form_2_u_p_isoparametric_reference_data {
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

static const KernelDiagnostics navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa",
    "HEX27",
    3,
    64,
    27,
    16,
    4,
    32,
    52,
    1,
    0,
    0,
    0,
    0,
    0,
    36,
    11,
    92,
    0,
    0,
    7,
    21,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa",
    "HEX27",
    3,
    64,
    27,
    16,
    4,
    29,
    60,
    1,
    0,
    0,
    0,
    0,
    0,
    33,
    21,
    97,
    0,
    0,
    17,
    26,
    10,
    40,
    4,
    0,
    0,
    89,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_mixed_impl(
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
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_mixed_impl(
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

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_mixed_impl(
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
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
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
    const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::hex8_shape_1d()};
    const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::hex8_grad_1d()};
    const idx_t *const SFEM_RESTRICT field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
            const int stream = 81 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 89; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }
        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[DIM * DIM];
        for (int component = 0; component < DIM * DIM; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::q_weight_1d(), block_direction, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
                const int stream = 27 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
                const int stream = 54 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
                const int stream = 81 + local_shape;
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

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa(
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
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float(
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
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
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
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_grad_1d();
    const idx_t *const SFEM_RESTRICT field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
    const idx_t *const SFEM_RESTRICT coordinate_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
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
            const idx_t *const SFEM_RESTRICT element_shape = coordinate_elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
            const int stream = 81 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 89; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex8_shape_1d()};
        const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex8_grad_1d()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::q_weight_1d(), block_direction, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
                const int stream = 27 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
                const int stream = 54 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
                const int stream = 81 + local_shape;
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

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE void navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t element,
        const ptrdiff_t out_stride,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_ROW_STREAMS = 81;
    static constexpr int N_COL_STREAMS = 8;
    static constexpr int ROW_COMPONENT[N_ROW_STREAMS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    static constexpr int ROW_SHAPE[N_ROW_STREAMS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
    static constexpr int COL_COMPONENT[N_COL_STREAMS] = {3, 3, 3, 3, 3, 3, 3, 3};
    static constexpr int COL_SHAPE[N_COL_STREAMS] = {0, 1, 2, 3, 4, 5, 6, 7};
    const ptrdiff_t element_offset = element * N_ROW_STREAMS * N_COL_STREAMS;
    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {
        const idx_t row_node = elements[ROW_SHAPE[row_stream]][element];
        const idx_t global_row = row_node * out_stride + ROW_COMPONENT[row_stream];
        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {
            const idx_t col_node = elements[COL_SHAPE[col_stream]][element];
            const ptrdiff_t entry = element_offset + row_stream * N_COL_STREAMS + col_stream;
            rows[entry] = global_row;
            cols[entry] = col_node * out_stride + COL_COMPONENT[col_stream];
            values[entry] = element_matrix[row_stream * N_COL_STREAMS + col_stream];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_grad_1d();
    const idx_t *const SFEM_RESTRICT field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
    const idx_t *const SFEM_RESTRICT coordinate_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t element_matrix[648];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = coordinate_elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex8_shape_1d()};
        const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::hex8_grad_1d()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        const auto row_tensor_stream = [](const int local) -> int {
            switch (local) {
                case 0: return 0;
                case 1: return 2;
                case 2: return 8;
                case 3: return 6;
                case 4: return 18;
                case 5: return 20;
                case 6: return 26;
                case 7: return 24;
                case 8: return 1;
                case 9: return 5;
                case 10: return 7;
                case 11: return 3;
                case 12: return 19;
                case 13: return 23;
                case 14: return 25;
                case 15: return 21;
                case 16: return 9;
                case 17: return 11;
                case 18: return 17;
                case 19: return 15;
                case 20: return 10;
                case 21: return 14;
                case 22: return 16;
                case 23: return 12;
                case 24: return 4;
                case 25: return 22;
                case 26: return 13;
                case 27: return 27;
                case 28: return 29;
                case 29: return 35;
                case 30: return 33;
                case 31: return 45;
                case 32: return 47;
                case 33: return 53;
                case 34: return 51;
                case 35: return 28;
                case 36: return 32;
                case 37: return 34;
                case 38: return 30;
                case 39: return 46;
                case 40: return 50;
                case 41: return 52;
                case 42: return 48;
                case 43: return 36;
                case 44: return 38;
                case 45: return 44;
                case 46: return 42;
                case 47: return 37;
                case 48: return 41;
                case 49: return 43;
                case 50: return 39;
                case 51: return 31;
                case 52: return 49;
                case 53: return 40;
                case 54: return 54;
                case 55: return 56;
                case 56: return 62;
                case 57: return 60;
                case 58: return 72;
                case 59: return 74;
                case 60: return 80;
                case 61: return 78;
                case 62: return 55;
                case 63: return 59;
                case 64: return 61;
                case 65: return 57;
                case 66: return 73;
                case 67: return 77;
                case 68: return 79;
                case 69: return 75;
                case 70: return 63;
                case 71: return 65;
                case 72: return 71;
                case 73: return 69;
                case 74: return 64;
                case 75: return 68;
                case 76: return 70;
                case 77: return 66;
                case 78: return 58;
                case 79: return 76;
                case 80: return 67;
                default: return 0;
            }
        };
        const auto col_tensor_stream = [](const int local) -> int {
            switch (local) {
                case 0: return 81;
                case 1: return 82;
                case 2: return 84;
                case 3: return 83;
                case 4: return 85;
                case 5: return 86;
                case 6: return 88;
                case 7: return 87;
                default: return 0;
            }
        };
        for (int entry = 0; entry < 648; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        for (int trial_local = 0; trial_local < 8; ++trial_local) {
            const int trial = col_tensor_stream(trial_local);
            for (int stream = 0; stream < N_FIELD_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[trial][0] = scalar_t(1);
            navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_determinant, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::q_weight_1d(), block_direction, block_output);
            for (int test_local = 0; test_local < 81; ++test_local) {
                const int test = row_tensor_stream(test_local);
                element_matrix[test_local * 8 + trial_local] = block_output[test][0];
            }
        }

        navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(elements, element, out_stride, element_matrix, rows, cols, values);
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, rows, cols, values);
}

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, rows, cols, values);
}
