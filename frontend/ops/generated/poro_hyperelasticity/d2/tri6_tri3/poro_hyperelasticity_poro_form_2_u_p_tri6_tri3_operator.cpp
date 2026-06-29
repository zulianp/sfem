#include "../poro_hyperelasticity_poro_form_2_u_p_d2_simplex_mixed_local.hpp"
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


template <typename scalar_t>
struct poro_hyperelasticity_poro_form_2_u_p_affine_reference_data {
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
struct poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data {
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

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    7,
    16,
    1,
    0,
    0,
    0,
    0,
    0,
    17,
    4,
    31,
    1,
    15,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    4,
    15,
    1,
    0,
    0,
    0,
    0,
    0,
    14,
    6,
    27,
    3,
    11,
    5,
    162,
    6,
    1,
    0,
    15,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t alpha,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[2],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri6_shape(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri3_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri6_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri6_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri3_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::tri3_grad_ref_y()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
        }

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_direction[0][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[0][lane] = scalar_t(0);
            block_direction[1][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[1][lane] = scalar_t(0);
            block_direction[2][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[2][lane] = scalar_t(0);
            block_direction[3][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_output[3][lane] = scalar_t(0);
            block_direction[4][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_output[4][lane] = scalar_t(0);
            block_direction[5][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_output[5][lane] = scalar_t(0);
            block_direction[6][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[6][lane] = scalar_t(0);
            block_direction[7][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[7][lane] = scalar_t(0);
            block_direction[8][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[8][lane] = scalar_t(0);
            block_direction[9][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_output[9][lane] = scalar_t(0);
            block_direction[10][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_output[10][lane] = scalar_t(0);
            block_direction[11][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_output[11][lane] = scalar_t(0);
            block_direction[12][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[12][lane] = scalar_t(0);
            block_direction[13][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[13][lane] = scalar_t(0);
            block_direction[14][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[14][lane] = scalar_t(0);
        }
        const scalar_t *const block_adjugate[DIM * DIM] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14]};

        poro_hyperelasticity_poro_form_2_u_p_d2_simplex_mixed_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_determinant0 + evbegin, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_affine_reference_data<scalar_t>::q_weight(), block_direction_streams, alpha, block_output_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[0][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[1][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[2][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 3] * out_stride] += block_output[3][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 4] * out_stride] += block_output[4][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 5] * out_stride] += block_output[5][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[6][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[7][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[8][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 3] * out_stride] += block_output[9][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 4] * out_stride] += block_output[10][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 5] * out_stride] += block_output[11][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[12][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[13][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[14][lane];
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, alpha, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, alpha, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t alpha,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[2],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[2],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 6;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
        }

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[2][lane] = points[0][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[3][lane] = points[1][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[4][lane] = points[0][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[5][lane] = points[1][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[6][lane] = points[0][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[7][lane] = points[1][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[8][lane] = points[0][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[9][lane] = points[1][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[10][lane] = points[0][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[11][lane] = points[1][ev[lane * CELL_N_SHAPE + 5]];
            block_direction[0][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[0][lane] = scalar_t(0);
            block_direction[1][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[1][lane] = scalar_t(0);
            block_direction[2][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[2][lane] = scalar_t(0);
            block_direction[3][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_output[3][lane] = scalar_t(0);
            block_direction[4][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_output[4][lane] = scalar_t(0);
            block_direction[5][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_output[5][lane] = scalar_t(0);
            block_direction[6][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[6][lane] = scalar_t(0);
            block_direction[7][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[7][lane] = scalar_t(0);
            block_direction[8][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[8][lane] = scalar_t(0);
            block_direction[9][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_output[9][lane] = scalar_t(0);
            block_direction[10][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_output[10][lane] = scalar_t(0);
            block_direction[11][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_output[11][lane] = scalar_t(0);
            block_direction[12][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_output[12][lane] = scalar_t(0);
            block_direction[13][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_output[13][lane] = scalar_t(0);
            block_direction[14][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_output[14][lane] = scalar_t(0);
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri6_shape(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri3_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri6_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri6_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_y()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14]};

        poro_hyperelasticity_poro_form_2_u_p_d2_simplex_mixed_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_isoparametric_reference_data<scalar_t>::q_weight(), block_direction_streams, alpha, block_output_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[0][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[1][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[2][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 3] * out_stride] += block_output[3][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 4] * out_stride] += block_output[4][lane];
#pragma omp atomic update
            u_out[0][ev[lane * CELL_N_SHAPE + 5] * out_stride] += block_output[5][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[6][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[7][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[8][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 3] * out_stride] += block_output[9][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 4] * out_stride] += block_output[10][lane];
#pragma omp atomic update
            u_out[1][ev[lane * CELL_N_SHAPE + 5] * out_stride] += block_output[11][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 0] * out_stride] += block_output[12][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 1] * out_stride] += block_output[13][lane];
#pragma omp atomic update
            p_out[ev[lane * CELL_N_SHAPE + 2] * out_stride] += block_output[14][lane];
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, alpha, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, alpha, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}
