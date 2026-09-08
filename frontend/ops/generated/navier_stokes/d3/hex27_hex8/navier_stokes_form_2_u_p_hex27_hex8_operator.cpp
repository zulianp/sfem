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
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
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
struct navier_stokes_form_2_u_p_affine_reference_data {
    static const s_t *q_weight_1d() {
        static const s_t data[4] = {s_t(0.17392742256872692), s_t(0.3260725774312731), s_t(0.3260725774312731), s_t(0.17392742256872692)};
        return data;
    }
    static const s_t *hex27_shape_1d() {
        static const s_t data[12] = {s_t(0.80134602936993082), s_t(0.25844425285419081), s_t(-0.059790282224121687), s_t(0.22778407679095203), s_t(0.88441289000295209), s_t(-0.11219696679390417), s_t(-0.11219696679390401), s_t(0.88441289000295198), s_t(0.22778407679095214), s_t(-0.05979028222412186), s_t(0.25844425285419081), s_t(0.80134602936993082)};
        return data;
    }
    static const s_t *hex27_grad_1d() {
        static const s_t data[12] = {s_t(-2.7222726231881049), s_t(3.4445452463762103), s_t(-0.72227262318810515), s_t(-1.6799620871697125), s_t(1.359924174339425), s_t(0.32003791283028749), s_t(-0.32003791283028749), s_t(-1.359924174339425), s_t(1.6799620871697125), s_t(0.72227262318810492), s_t(-3.4445452463762098), s_t(2.7222726231881049)};
        return data;
    }
    static const s_t *hex8_shape_1d() {
        static const s_t data[8] = {s_t(0.93056815579702623), s_t(0.069431844202973714), s_t(0.66999052179242813), s_t(0.33000947820757187), s_t(0.33000947820757187), s_t(0.66999052179242813), s_t(0.069431844202973769), s_t(0.93056815579702623)};
        return data;
    }
    static const s_t *hex8_grad_1d() {
        static const s_t data[8] = {s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
};

template <typename s_t>
struct navier_stokes_form_2_u_p_isoparametric_reference_data {
    static const s_t *q_weight_1d() {
        static const s_t data[4] = {s_t(0.17392742256872692), s_t(0.3260725774312731), s_t(0.3260725774312731), s_t(0.17392742256872692)};
        return data;
    }
    static const s_t *hex27_shape_1d() {
        static const s_t data[12] = {s_t(0.80134602936993082), s_t(0.25844425285419081), s_t(-0.059790282224121687), s_t(0.22778407679095203), s_t(0.88441289000295209), s_t(-0.11219696679390417), s_t(-0.11219696679390401), s_t(0.88441289000295198), s_t(0.22778407679095214), s_t(-0.05979028222412186), s_t(0.25844425285419081), s_t(0.80134602936993082)};
        return data;
    }
    static const s_t *hex27_grad_1d() {
        static const s_t data[12] = {s_t(-2.7222726231881049), s_t(3.4445452463762103), s_t(-0.72227262318810515), s_t(-1.6799620871697125), s_t(1.359924174339425), s_t(0.32003791283028749), s_t(-0.32003791283028749), s_t(-1.359924174339425), s_t(1.6799620871697125), s_t(0.72227262318810492), s_t(-3.4445452463762098), s_t(2.7222726231881049)};
        return data;
    }
    static const s_t *hex8_shape_1d() {
        static const s_t data[8] = {s_t(0.93056815579702623), s_t(0.069431844202973714), s_t(0.66999052179242813), s_t(0.33000947820757187), s_t(0.33000947820757187), s_t(0.66999052179242813), s_t(0.069431844202973769), s_t(0.93056815579702623)};
        return data;
    }
    static const s_t *hex8_grad_1d() {
        static const s_t data[8] = {s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1)};
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
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
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
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 64;
    static constexpr int CELL_NS = 27;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 64;
    static constexpr int CELL_NS = 27;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u_direction_data[3],
        const s_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 64;
    static constexpr int CELL_NS = 27;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const field_shape_1d[NC] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<s_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<s_t>::hex8_shape_1d()};
    const s_t *const field_grad_1d[NC] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<s_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<s_t>::hex8_grad_1d()};
    const idx_t *const SFEM_RESTRICT field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bdirection[N_FIELD_STREAMS][VS];
        s_t boutput[N_FIELD_STREAMS][VS];

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
            const int stream = 81 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 89; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }
        const g_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_adjugate4 + evb, g_jacobian_adjugate5 + evb, g_jacobian_adjugate6 + evb, g_jacobian_adjugate7 + evb, g_jacobian_adjugate8 + evb, g_jacobian_determinant0 + evb};
        s_t baffine_geometry_data[10][VS];
        const s_t *bageom_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *badjugate[ND * ND];
        for (int component = 0; component < ND * ND; ++component) {
            badjugate[component] = bageom_streams[component];
        }

        navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, 0, bageom_streams[9], badjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<s_t>::q_weight_1d(), bdirection, boutput);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
                const int stream = 27 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
                const int stream = 54 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
                const int stream = 81 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
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

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u_direction_data[3],
        const s_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 64;
    static constexpr int CELL_NS = 27;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_shape_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex27_shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex27_grad_1d();
    const idx_t *const SFEM_RESTRICT field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
    const idx_t *const SFEM_RESTRICT field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
    const idx_t *const SFEM_RESTRICT coordinate_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bcoordinates[ND * CELL_NS][VS];
        s_t badjugate_data[ND * ND][NQ * VS];
        s_t bdeterminant[NQ * VS];
        s_t bdirection[N_FIELD_STREAMS][VS];
        s_t boutput[N_FIELD_STREAMS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = coordinate_elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }

        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
            const int stream = 27 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 27; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
            const int stream = 54 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
            const int stream = 81 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                bdirection[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 89; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
                coordinate_grad_ref + 2 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

        const s_t *const field_shape_1d[NC] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex8_shape_1d()};
        const s_t *const field_grad_1d[NC] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::hex8_grad_1d()};
        const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

        navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, VS, bdeterminant, badjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<s_t>::q_weight_1d(), bdirection, boutput);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_0_elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_1_elements[local_shape];
                const int stream = 27 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 27; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_2_elements[local_shape];
                const int stream = 54 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = field_3_elements[local_shape];
                const int stream = 81 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
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
