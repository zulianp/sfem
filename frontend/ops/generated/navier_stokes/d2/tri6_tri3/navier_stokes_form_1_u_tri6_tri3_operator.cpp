#include <type_traits>
#include "../navier_stokes_form_1_u_d2_simplex_mixed_local.hpp"
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
struct navier_stokes_form_1_u_affine_reference_data {
    static const s_t *q_weight() {
        static const s_t data[6] = {s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.054975871827660998), s_t(0.054975871827660998), s_t(0.054975871827660998)};
        return data;
    }
    static const s_t *tri6_shape() {
        static const s_t data[36] = {s_t(-0.084730493093977982), s_t(-0.04820837781551205), s_t(-0.04820837781551205), s_t(0.1928335112620482), s_t(0.79548022620090564), s_t(0.1928335112620482), s_t(-0.04820837781551205), s_t(-0.084730493093977968), s_t(-0.04820837781551205), s_t(0.19283351126204817), s_t(0.19283351126204817), s_t(0.79548022620090564), s_t(-0.04820837781551205), s_t(-0.04820837781551205), s_t(-0.084730493093977968), s_t(0.79548022620090564), s_t(0.19283351126204817), s_t(0.19283351126204817), s_t(0.5176323419876725), s_t(-0.074803807748196505), s_t(-0.074803807748196505), s_t(0.29921523099278602), s_t(0.03354481152314847), s_t(0.29921523099278602), s_t(-0.074803807748196505), s_t(0.5176323419876725), s_t(-0.074803807748196505), s_t(0.29921523099278602), s_t(0.29921523099278602), s_t(0.03354481152314847), s_t(-0.074803807748196505), s_t(-0.074803807748196505), s_t(0.5176323419876725), s_t(0.03354481152314847), s_t(0.29921523099278602), s_t(0.29921523099278602)};
        return data;
    }
    static const s_t *tri6_grad_ref_x() {
        static const s_t data[36] = {s_t(0.56758792732771912), s_t(0.78379396366385956), s_t(0), s_t(-1.3513818909915787), s_t(1.7837939636638596), s_t(-1.7837939636638596), s_t(-0.78379396366385956), s_t(-0.56758792732771912), s_t(0), s_t(1.3513818909915787), s_t(1.7837939636638596), s_t(-1.7837939636638596), s_t(-0.78379396366385956), s_t(0.78379396366385956), s_t(0), s_t(5.5511151231257827e-17), s_t(0.43241207267228082), s_t(-0.43241207267228082), s_t(-2.2673902919218341), s_t(-0.63369514596091703), s_t(0), s_t(2.9010854378827511), s_t(0.36630485403908297), s_t(-0.36630485403908297), s_t(0.63369514596091703), s_t(2.2673902919218341), s_t(0), s_t(-2.9010854378827511), s_t(0.36630485403908297), s_t(-0.36630485403908297), s_t(0.63369514596091703), s_t(-0.63369514596091703), s_t(0), s_t(0), s_t(3.2673902919218341), s_t(-3.2673902919218341)};
        return data;
    }
    static const s_t *tri6_grad_ref_y() {
        static const s_t data[36] = {s_t(0.56758792732771912), s_t(0), s_t(0.78379396366385956), s_t(-1.7837939636638596), s_t(1.7837939636638596), s_t(-1.3513818909915787), s_t(-0.78379396366385956), s_t(0), s_t(0.78379396366385956), s_t(-0.43241207267228082), s_t(0.43241207267228082), s_t(0), s_t(-0.78379396366385956), s_t(0), s_t(-0.56758792732771912), s_t(-1.7837939636638596), s_t(1.7837939636638596), s_t(1.3513818909915787), s_t(-2.2673902919218341), s_t(0), s_t(-0.63369514596091703), s_t(-0.36630485403908297), s_t(0.36630485403908297), s_t(2.9010854378827511), s_t(0.63369514596091703), s_t(0), s_t(-0.63369514596091703), s_t(-3.2673902919218341), s_t(3.2673902919218341), s_t(0), s_t(0.63369514596091703), s_t(0), s_t(2.2673902919218341), s_t(-0.36630485403908297), s_t(0.36630485403908297), s_t(-2.9010854378827511)};
        return data;
    }
    static const s_t *tri3_shape() {
        static const s_t data[18] = {s_t(0.10810301816807022), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851)};
        return data;
    }
    static const s_t *tri3_grad_ref_x() {
        static const s_t data[18] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *tri3_grad_ref_y() {
        static const s_t data[18] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
        return data;
    }
};

template <typename s_t>
struct navier_stokes_form_1_u_isoparametric_reference_data {
    static const s_t *q_weight() {
        static const s_t data[6] = {s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.054975871827660998), s_t(0.054975871827660998), s_t(0.054975871827660998)};
        return data;
    }
    static const s_t *tri6_shape() {
        static const s_t data[36] = {s_t(-0.084730493093977982), s_t(-0.04820837781551205), s_t(-0.04820837781551205), s_t(0.1928335112620482), s_t(0.79548022620090564), s_t(0.1928335112620482), s_t(-0.04820837781551205), s_t(-0.084730493093977968), s_t(-0.04820837781551205), s_t(0.19283351126204817), s_t(0.19283351126204817), s_t(0.79548022620090564), s_t(-0.04820837781551205), s_t(-0.04820837781551205), s_t(-0.084730493093977968), s_t(0.79548022620090564), s_t(0.19283351126204817), s_t(0.19283351126204817), s_t(0.5176323419876725), s_t(-0.074803807748196505), s_t(-0.074803807748196505), s_t(0.29921523099278602), s_t(0.03354481152314847), s_t(0.29921523099278602), s_t(-0.074803807748196505), s_t(0.5176323419876725), s_t(-0.074803807748196505), s_t(0.29921523099278602), s_t(0.29921523099278602), s_t(0.03354481152314847), s_t(-0.074803807748196505), s_t(-0.074803807748196505), s_t(0.5176323419876725), s_t(0.03354481152314847), s_t(0.29921523099278602), s_t(0.29921523099278602)};
        return data;
    }
    static const s_t *tri6_grad_ref_x() {
        static const s_t data[36] = {s_t(0.56758792732771912), s_t(0.78379396366385956), s_t(0), s_t(-1.3513818909915787), s_t(1.7837939636638596), s_t(-1.7837939636638596), s_t(-0.78379396366385956), s_t(-0.56758792732771912), s_t(0), s_t(1.3513818909915787), s_t(1.7837939636638596), s_t(-1.7837939636638596), s_t(-0.78379396366385956), s_t(0.78379396366385956), s_t(0), s_t(5.5511151231257827e-17), s_t(0.43241207267228082), s_t(-0.43241207267228082), s_t(-2.2673902919218341), s_t(-0.63369514596091703), s_t(0), s_t(2.9010854378827511), s_t(0.36630485403908297), s_t(-0.36630485403908297), s_t(0.63369514596091703), s_t(2.2673902919218341), s_t(0), s_t(-2.9010854378827511), s_t(0.36630485403908297), s_t(-0.36630485403908297), s_t(0.63369514596091703), s_t(-0.63369514596091703), s_t(0), s_t(0), s_t(3.2673902919218341), s_t(-3.2673902919218341)};
        return data;
    }
    static const s_t *tri6_grad_ref_y() {
        static const s_t data[36] = {s_t(0.56758792732771912), s_t(0), s_t(0.78379396366385956), s_t(-1.7837939636638596), s_t(1.7837939636638596), s_t(-1.3513818909915787), s_t(-0.78379396366385956), s_t(0), s_t(0.78379396366385956), s_t(-0.43241207267228082), s_t(0.43241207267228082), s_t(0), s_t(-0.78379396366385956), s_t(0), s_t(-0.56758792732771912), s_t(-1.7837939636638596), s_t(1.7837939636638596), s_t(1.3513818909915787), s_t(-2.2673902919218341), s_t(0), s_t(-0.63369514596091703), s_t(-0.36630485403908297), s_t(0.36630485403908297), s_t(2.9010854378827511), s_t(0.63369514596091703), s_t(0), s_t(-0.63369514596091703), s_t(-3.2673902919218341), s_t(3.2673902919218341), s_t(0), s_t(0.63369514596091703), s_t(0), s_t(2.2673902919218341), s_t(-0.36630485403908297), s_t(0.36630485403908297), s_t(-2.9010854378827511)};
        return data;
    }
    static const s_t *tri3_shape() {
        static const s_t data[18] = {s_t(0.10810301816807022), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851)};
        return data;
    }
    static const s_t *tri3_grad_ref_x() {
        static const s_t data[18] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *tri3_grad_ref_y() {
        static const s_t data[18] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_1_u_tri6_tri3_residual_element_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    16,
    31,
    1,
    0,
    0,
    0,
    0,
    0,
    22,
    8,
    55,
    0,
    0,
    5,
    16,
    5,
    162,
    6,
    6,
    30,
    0,
    15,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_u_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_u_tri6_tri3_residual_element_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_u_tri6_tri3_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    13,
    31,
    1,
    0,
    0,
    0,
    0,
    0,
    20,
    11,
    52,
    0,
    0,
    8,
    16,
    5,
    162,
    6,
    0,
    0,
    0,
    15,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t convection_scale,
        const s_t dt,
        const s_t f0,
        const s_t f1,
        const s_t nu,
        const s_t rho,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u_data[2],
        const s_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u_old_data[2],
        const s_t *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[2],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 6;
    static constexpr int CELL_NS = 6;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const field_shape[NC] = {sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri6_shape(), sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri3_shape()};
    const s_t *const field_grad_ref[NC * ND] = {sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri6_grad_ref_x(), sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri6_grad_ref_y(), sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri3_grad_ref_x(), sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::tri3_grad_ref_y()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        s_t block_current[N_FIELD_STREAMS][VS];
        s_t block_previous[N_FIELD_STREAMS][VS];
        s_t block_output[N_FIELD_STREAMS][VS];

        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = u_data[0][node * current_stride];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
            }
        }
        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 6 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = u_data[1][node * current_stride];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
            }
        }
        for (int local_shape = 0; local_shape < 3; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 12 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = p_data[node * current_stride];
                block_previous[stream][lane] = p_old_data[node * previous_stride];
            }
        }

        for (int stream = 0; stream < 15; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }
        const g_t *const affine_geometry_sources[5] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin};
        s_t block_affine_geometry_data[5][VS];
        const s_t *block_affine_geometry_streams[5];
        for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *block_adjugate[ND * ND];
        for (int component = 0; component < ND * ND; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        navier_stokes_form_1_u_d2_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, 0, block_affine_geometry_streams[4], block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_1_u_affine_reference_data<s_t>::q_weight(), block_current, block_previous, convection_scale, dt, f0, f1, nu, rho, block_output);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
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
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 6 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 3; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 12 + local_shape;
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

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa(
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
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_float(
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
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t convection_scale,
        const s_t dt,
        const s_t f0,
        const s_t f1,
        const s_t nu,
        const s_t rho,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u_data[2],
        const s_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u_old_data[2],
        const s_t *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[2],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 6;
    static constexpr int CELL_NS = 6;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri6_grad_ref_x();
    const s_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri6_grad_ref_y();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        s_t block_coordinates[ND * CELL_NS][VS];
        s_t block_adjugate_data[ND * ND][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[N_FIELD_STREAMS][VS];
        s_t block_previous[N_FIELD_STREAMS][VS];
        s_t block_output[N_FIELD_STREAMS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }

        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = u_data[0][node * current_stride];
                block_previous[stream][lane] = u_old_data[0][node * previous_stride];
            }
        }
        for (int local_shape = 0; local_shape < 6; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 6 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = u_data[1][node * current_stride];
                block_previous[stream][lane] = u_old_data[1][node * previous_stride];
            }
        }
        for (int local_shape = 0; local_shape < 3; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 12 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_current[stream][lane] = p_data[node * current_stride];
                block_previous[stream][lane] = p_old_data[node * previous_stride];
            }
        }

        for (int stream = 0; stream < 15; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t *block_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5];
                const s_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5];
                const s_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5];
                const s_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VS + lane);
            }
        }

        const s_t *const field_shape[NC] = {sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri6_shape(), sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri3_shape()};
        const s_t *const field_grad_ref[NC * ND] = {sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri6_grad_ref_x(), sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri6_grad_ref_y(), sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri3_grad_ref_x(), sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::tri3_grad_ref_y()};
        const s_t *const block_adjugate[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

        navier_stokes_form_1_u_d2_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, VS, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_1_u_isoparametric_reference_data<s_t>::q_weight(), block_current, block_previous, convection_scale, dt, f0, f1, nu, rho, block_output);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
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
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 6; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 6 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 3; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 12 + local_shape;
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

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[2],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 6;
    static constexpr int CELL_NS = 6;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[2],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 6;
    static constexpr int CELL_NS = 6;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
