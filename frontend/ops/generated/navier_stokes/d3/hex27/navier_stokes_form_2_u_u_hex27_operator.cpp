#include <type_traits>
#include "../navier_stokes_form_2_u_u_d3_tensor_product_local.hpp"
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
struct navier_stokes_form_2_u_u_affine_reference_data {
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
};

template <typename scalar_t>
struct navier_stokes_form_2_u_u_isoparametric_reference_data {
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
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_u_hex27_residual_element_soa",
    "HEX27",
    3,
    64,
    27,
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
    3,
    0,
    0,
    0,
    0,
    0,
    10,
    24,
    4,
    0,
    0,
    0,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_u_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_hex27_residual_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_hex27_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa",
    "HEX27",
    3,
    64,
    27,
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
    3,
    0,
    0,
    0,
    0,
    0,
    10,
    24,
    4,
    4,
    81,
    81,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_hex27_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3]
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3]
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_mixed_impl(
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
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3]
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::hex27_shape_1d()};
    const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::hex27_grad_1d()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[2][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
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
        const scalar_t *const block_previous_streams[81] = {block_previous[0], block_previous[8], block_previous[1], block_previous[11], block_previous[24], block_previous[9], block_previous[3], block_previous[10], block_previous[2], block_previous[16], block_previous[20], block_previous[17], block_previous[23], block_previous[26], block_previous[21], block_previous[19], block_previous[22], block_previous[18], block_previous[4], block_previous[12], block_previous[5], block_previous[15], block_previous[25], block_previous[13], block_previous[7], block_previous[14], block_previous[6], block_previous[27], block_previous[35], block_previous[28], block_previous[38], block_previous[51], block_previous[36], block_previous[30], block_previous[37], block_previous[29], block_previous[43], block_previous[47], block_previous[44], block_previous[50], block_previous[53], block_previous[48], block_previous[46], block_previous[49], block_previous[45], block_previous[31], block_previous[39], block_previous[32], block_previous[42], block_previous[52], block_previous[40], block_previous[34], block_previous[41], block_previous[33], block_previous[54], block_previous[62], block_previous[55], block_previous[65], block_previous[78], block_previous[63], block_previous[57], block_previous[64], block_previous[56], block_previous[70], block_previous[74], block_previous[71], block_previous[77], block_previous[80], block_previous[75], block_previous[73], block_previous[76], block_previous[72], block_previous[58], block_previous[66], block_previous[59], block_previous[69], block_previous[79], block_previous[67], block_previous[61], block_previous[68], block_previous[60]};
        const scalar_t *const block_direction_streams[81] = {block_direction[0], block_direction[8], block_direction[1], block_direction[11], block_direction[24], block_direction[9], block_direction[3], block_direction[10], block_direction[2], block_direction[16], block_direction[20], block_direction[17], block_direction[23], block_direction[26], block_direction[21], block_direction[19], block_direction[22], block_direction[18], block_direction[4], block_direction[12], block_direction[5], block_direction[15], block_direction[25], block_direction[13], block_direction[7], block_direction[14], block_direction[6], block_direction[27], block_direction[35], block_direction[28], block_direction[38], block_direction[51], block_direction[36], block_direction[30], block_direction[37], block_direction[29], block_direction[43], block_direction[47], block_direction[44], block_direction[50], block_direction[53], block_direction[48], block_direction[46], block_direction[49], block_direction[45], block_direction[31], block_direction[39], block_direction[32], block_direction[42], block_direction[52], block_direction[40], block_direction[34], block_direction[41], block_direction[33], block_direction[54], block_direction[62], block_direction[55], block_direction[65], block_direction[78], block_direction[63], block_direction[57], block_direction[64], block_direction[56], block_direction[70], block_direction[74], block_direction[71], block_direction[77], block_direction[80], block_direction[75], block_direction[73], block_direction[76], block_direction[72], block_direction[58], block_direction[66], block_direction[59], block_direction[69], block_direction[79], block_direction[67], block_direction[61], block_direction[68], block_direction[60]};
        scalar_t *const block_output_streams[81] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6], block_output[27], block_output[35], block_output[28], block_output[38], block_output[51], block_output[36], block_output[30], block_output[37], block_output[29], block_output[43], block_output[47], block_output[44], block_output[50], block_output[53], block_output[48], block_output[46], block_output[49], block_output[45], block_output[31], block_output[39], block_output[32], block_output[42], block_output[52], block_output[40], block_output[34], block_output[41], block_output[33], block_output[54], block_output[62], block_output[55], block_output[65], block_output[78], block_output[63], block_output[57], block_output[64], block_output[56], block_output[70], block_output[74], block_output[71], block_output[77], block_output[80], block_output[75], block_output[73], block_output[76], block_output[72], block_output[58], block_output[66], block_output[59], block_output[69], block_output[79], block_output[67], block_output[61], block_output[68], block_output[60]};

        navier_stokes_form_2_u_u_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::q_weight_1d(), block_previous_streams, block_direction_streams, convection_scale, dt, nu, rho, block_output_streams);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
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
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
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
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 54 + local_shape;
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

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3]
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 27;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::hex27_shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::hex27_grad_1d();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
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

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[2][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
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

        const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::hex27_shape_1d()};
        const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::hex27_grad_1d()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_previous_streams[81] = {block_previous[0], block_previous[8], block_previous[1], block_previous[11], block_previous[24], block_previous[9], block_previous[3], block_previous[10], block_previous[2], block_previous[16], block_previous[20], block_previous[17], block_previous[23], block_previous[26], block_previous[21], block_previous[19], block_previous[22], block_previous[18], block_previous[4], block_previous[12], block_previous[5], block_previous[15], block_previous[25], block_previous[13], block_previous[7], block_previous[14], block_previous[6], block_previous[27], block_previous[35], block_previous[28], block_previous[38], block_previous[51], block_previous[36], block_previous[30], block_previous[37], block_previous[29], block_previous[43], block_previous[47], block_previous[44], block_previous[50], block_previous[53], block_previous[48], block_previous[46], block_previous[49], block_previous[45], block_previous[31], block_previous[39], block_previous[32], block_previous[42], block_previous[52], block_previous[40], block_previous[34], block_previous[41], block_previous[33], block_previous[54], block_previous[62], block_previous[55], block_previous[65], block_previous[78], block_previous[63], block_previous[57], block_previous[64], block_previous[56], block_previous[70], block_previous[74], block_previous[71], block_previous[77], block_previous[80], block_previous[75], block_previous[73], block_previous[76], block_previous[72], block_previous[58], block_previous[66], block_previous[59], block_previous[69], block_previous[79], block_previous[67], block_previous[61], block_previous[68], block_previous[60]};
        const scalar_t *const block_direction_streams[81] = {block_direction[0], block_direction[8], block_direction[1], block_direction[11], block_direction[24], block_direction[9], block_direction[3], block_direction[10], block_direction[2], block_direction[16], block_direction[20], block_direction[17], block_direction[23], block_direction[26], block_direction[21], block_direction[19], block_direction[22], block_direction[18], block_direction[4], block_direction[12], block_direction[5], block_direction[15], block_direction[25], block_direction[13], block_direction[7], block_direction[14], block_direction[6], block_direction[27], block_direction[35], block_direction[28], block_direction[38], block_direction[51], block_direction[36], block_direction[30], block_direction[37], block_direction[29], block_direction[43], block_direction[47], block_direction[44], block_direction[50], block_direction[53], block_direction[48], block_direction[46], block_direction[49], block_direction[45], block_direction[31], block_direction[39], block_direction[32], block_direction[42], block_direction[52], block_direction[40], block_direction[34], block_direction[41], block_direction[33], block_direction[54], block_direction[62], block_direction[55], block_direction[65], block_direction[78], block_direction[63], block_direction[57], block_direction[64], block_direction[56], block_direction[70], block_direction[74], block_direction[71], block_direction[77], block_direction[80], block_direction[75], block_direction[73], block_direction[76], block_direction[72], block_direction[58], block_direction[66], block_direction[59], block_direction[69], block_direction[79], block_direction[67], block_direction[61], block_direction[68], block_direction[60]};
        scalar_t *const block_output_streams[81] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6], block_output[27], block_output[35], block_output[28], block_output[38], block_output[51], block_output[36], block_output[30], block_output[37], block_output[29], block_output[43], block_output[47], block_output[44], block_output[50], block_output[53], block_output[48], block_output[46], block_output[49], block_output[45], block_output[31], block_output[39], block_output[32], block_output[42], block_output[52], block_output[40], block_output[34], block_output[41], block_output[33], block_output[54], block_output[62], block_output[55], block_output[65], block_output[78], block_output[63], block_output[57], block_output[64], block_output[56], block_output[70], block_output[74], block_output[71], block_output[77], block_output[80], block_output[75], block_output[73], block_output[76], block_output[72], block_output[58], block_output[66], block_output[59], block_output[69], block_output[79], block_output[67], block_output[61], block_output[68], block_output[60]};

        navier_stokes_form_2_u_u_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::q_weight_1d(), block_previous_streams, block_direction_streams, convection_scale, dt, nu, rho, block_output_streams);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
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
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
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
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 54 + local_shape;
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

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}
