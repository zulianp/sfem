#include <type_traits>
#include "../navier_stokes_form_2_u_u_d2_simplex_local.hpp"
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
struct navier_stokes_form_2_u_u_affine_reference_data {
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
    static const scalar_t *tri6_shape() {
        static const scalar_t data[36] = {scalar_t(-0.084730493093977982), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(0.1928335112620482), scalar_t(0.79548022620090564), scalar_t(0.1928335112620482), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(-0.04820837781551205), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.79548022620090564), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(0.79548022620090564), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602)};
        return data;
    }
    static const scalar_t *tri6_grad_ref_x() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
        return data;
    }
    static const scalar_t *tri6_grad_ref_y() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
        return data;
    }
};

template <typename scalar_t>
struct navier_stokes_form_2_u_u_isoparametric_reference_data {
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
    static const scalar_t *tri6_shape() {
        static const scalar_t data[36] = {scalar_t(-0.084730493093977982), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(0.1928335112620482), scalar_t(0.79548022620090564), scalar_t(0.1928335112620482), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(-0.04820837781551205), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.79548022620090564), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(0.79548022620090564), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602)};
        return data;
    }
    static const scalar_t *tri6_grad_ref_x() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
        return data;
    }
    static const scalar_t *tri6_grad_ref_y() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_u_tri6_residual_element_soa",
    "TRI6",
    2,
    6,
    6,
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
    2,
    0,
    0,
    0,
    0,
    0,
    5,
    108,
    6,
    0,
    0,
    0,
    12,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_u_tri6_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_tri6_residual_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_tri6_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa",
    "TRI6",
    2,
    6,
    6,
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
    2,
    0,
    0,
    0,
    0,
    0,
    5,
    108,
    6,
    4,
    12,
    12,
    12,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_tri6_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2]
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 12;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2]
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 12;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2]
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 12;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::tri6_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::tri6_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::tri6_grad_ref_y()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 6 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }

        for (int stream = 0; stream < 12; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }
        const jacobian_t *const affine_geometry_sources[5] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[5][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[5];
        for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[DIM * DIM];
        for (int component = 0; component < DIM * DIM; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        navier_stokes_form_2_u_u_d2_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[4], block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_u_affine_reference_data<scalar_t>::q_weight(), block_previous, block_direction, convection_scale, dt, nu, rho, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
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
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 6 + local_shape;
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

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2]
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 12;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1]};
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

        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 6 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }

        for (int stream = 0; stream < 12; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

        navier_stokes_form_2_u_u_d2_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::q_weight(), block_previous, block_direction, convection_scale, dt, nu, rho, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
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
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 6 + local_shape;
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

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2]
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE void navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t element,
        const ptrdiff_t out_stride,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_ROW_STREAMS = 12;
    static constexpr int N_COL_STREAMS = 12;
    static constexpr int ROW_COMPONENT[N_ROW_STREAMS] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    static constexpr int ROW_SHAPE[N_ROW_STREAMS] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    static constexpr int COL_COMPONENT[N_COL_STREAMS] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    static constexpr int COL_SHAPE[N_COL_STREAMS] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
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
static SFEM_INLINE int navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 12;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t element_matrix[144];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1]};
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

        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
            }
        }
        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 6 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
            const int lane = 0;
            const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
            const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
            const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
            const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
            geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                    J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

        for (int entry = 0; entry < 144; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        for (int trial_local = 0; trial_local < 12; ++trial_local) {
            const int trial = trial_local;
            for (int stream = 0; stream < N_FIELD_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[trial][0] = scalar_t(1);
            navier_stokes_form_2_u_u_d2_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_u_isoparametric_reference_data<scalar_t>::q_weight(), block_previous, block_direction, convection_scale, dt, nu, rho, block_output);
            for (int test_local = 0; test_local < 12; ++test_local) {
                const int test = test_local;
                element_matrix[test_local * 12 + trial_local] = block_output[test][0];
            }
        }

        navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(elements, element, out_stride, element_matrix, rows, cols, values);
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, out_stride, rows, cols, values);
}

extern "C" int navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, out_stride, rows, cols, values);
}
