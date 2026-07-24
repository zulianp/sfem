#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../laplace_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>

#include "tet10_laplacian_inline_cpu.hpp"

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
struct laplace_tet10_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[40] = {scalar_t(0.099999999999999936), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.32360679774997891), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997886), scalar_t(0.32360679774997897), scalar_t(0.076393202250020983), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.32360679774997886), scalar_t(0.076393202250020983), scalar_t(0.076393202250021025), scalar_t(0.32360679774997897), scalar_t(-0.099999999999999964), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(0.07639320225002097), scalar_t(0.076393202250021025), scalar_t(0.07639320225002097), scalar_t(0.3236067977499788), scalar_t(0.32360679774997897), scalar_t(0.32360679774997897)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_tet10_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[40] = {scalar_t(0.099999999999999936), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.32360679774997891), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997886), scalar_t(0.32360679774997897), scalar_t(0.076393202250020983), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.32360679774997886), scalar_t(0.076393202250020983), scalar_t(0.076393202250021025), scalar_t(0.32360679774997897), scalar_t(-0.099999999999999964), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(0.07639320225002097), scalar_t(0.076393202250021025), scalar_t(0.07639320225002097), scalar_t(0.3236067977499788), scalar_t(0.32360679774997897), scalar_t(0.32360679774997897)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet10_residual_element_soa_diagnostics_data = {
    "laplace_tet10_residual_element_soa",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    160,
    4,
    1,
    10,
    0,
    10,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_tet10_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet10_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_residual_element_soa",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_residual_element_soa_float",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet10_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet10_residual_affine_mesh_soa",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet10_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet10_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet10_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet10_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet10_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet10_jacobian_u_u_diagnostics_data = {
    "laplace_tet10_jacobian_u_u",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    160,
    4,
    1,
    0,
    10,
    10,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_tet10_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_tet10_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet10_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet10_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_jacobian_u_u",
            &sfem::codegen::laplace_tet10_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_jacobian_u_u_float",
            &sfem::codegen::laplace_tet10_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet10_jacobian_action_element_soa_diagnostics_data = {
    "laplace_tet10_jacobian_action_element_soa",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    160,
    4,
    1,
    0,
    10,
    10,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_tet10_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet10_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_jacobian_action_element_soa",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet10_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet10_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet10_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet10_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet10_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet10_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet10_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet10_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet10_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_tet10_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[10],
        const double kappa,
        double *const SFEM_RESTRICT output[10]
) {
    sfem::codegen::laplace_d3_simplex_residual_block<double, 4, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::q_weight(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet10_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[10],
        const float kappa,
        float *const SFEM_RESTRICT output[10]
) {
    sfem::codegen::laplace_d3_simplex_residual_block<float, 4, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::q_weight(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet10_residual_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 10; ++stream) {
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
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        laplace_d3_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, block_current, kappa, block_output);

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

extern "C" int laplace_tet10_residual_affine_mesh_soa(
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
    return sfem::codegen::laplace_tet10_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet10_residual_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_tet10_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet10_residual_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 10; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current, kappa, block_output);

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

extern "C" int laplace_tet10_residual_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_tet10_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet10_residual_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_tet10_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet10_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_tet10_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_tet10_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_tet10_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_tet10_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[10],
        const double kappa,
        double *const SFEM_RESTRICT output[10]
) {
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<double, 4, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet10_isoparametric_reference_data<double>::q_weight(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet10_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[10],
        const float kappa,
        float *const SFEM_RESTRICT output[10]
) {
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<float, 4, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet10_isoparametric_reference_data<float>::q_weight(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet10_jacobian_action_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tet10_affine_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 10; ++stream) {
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
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, block_direction, kappa, block_output);

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

extern "C" int laplace_tet10_jacobian_action_affine_mesh_soa(
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
    return sfem::codegen::laplace_tet10_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_jacobian_action_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_tet10_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, bool UnitKappa>
static SFEM_INLINE int laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl_kernel(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int VECTOR_SIZE = 32;
    (void)nnodes;
    (void)max_nodes_per_pack;
    (void)direction_stride;
    (void)out_stride;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_direction = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)max_nodes_per_pack);
        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];
        scalar_t out4[VECTOR_SIZE];
        scalar_t out5[VECTOR_SIZE];
        scalar_t out6[VECTOR_SIZE];
        scalar_t out7[VECTOR_SIZE];
        scalar_t out8[VECTOR_SIZE];
        scalar_t out9[VECTOR_SIZE];
        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];
        scalar_t u4[VECTOR_SIZE];
        scalar_t u5[VECTOR_SIZE];
        scalar_t u6[VECTOR_SIZE];
        scalar_t u7[VECTOR_SIZE];
        scalar_t u8[VECTOR_SIZE];
        scalar_t u9[VECTOR_SIZE];
        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            scalar_t *const SFEM_RESTRICT ghost_out = &pack_out[n_contiguous];
            memcpy(pack_direction, &u_direction[owned_nodes_ptr[pack]], (size_t)n_contiguous * sizeof(scalar_t));
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                pack_direction[n_contiguous + k] = u_direction[ghosts[k]];
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                const scalar_t metric_factor = UnitKappa ? scalar_t(1) : kappa;

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    const ptrdiff_t metric_offset = element * 6;
                    fff0[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 0]);
                    fff1[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 1]);
                    fff2[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 2]);
                    fff3[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 3]);
                    fff4[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 4]);
                    fff5[lane] = metric_factor * scalar_t(g_geom_metric[metric_offset + 5]);
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    u0[lane] = pack_direction[elements[0][element]];
                    u1[lane] = pack_direction[elements[1][element]];
                    u2[lane] = pack_direction[elements[2][element]];
                    u3[lane] = pack_direction[elements[3][element]];
                    u4[lane] = pack_direction[elements[4][element]];
                    u5[lane] = pack_direction[elements[5][element]];
                    u6[lane] = pack_direction[elements[6][element]];
                    u7[lane] = pack_direction[elements[7][element]];
                    u8[lane] = pack_direction[elements[8][element]];
                    u9[lane] = pack_direction[elements[9][element]];
                }

#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    tet10_laplacian_apply_fff_soa_tpl<scalar_t, scalar_t>(fff0[lane], fff1[lane], fff2[lane], fff3[lane], fff4[lane], fff5[lane], u0[lane], u1[lane], u2[lane], u3[lane], u4[lane], u5[lane], u6[lane], u7[lane], u8[lane], u9[lane], &out0[lane], &out1[lane], &out2[lane], &out3[lane], &out4[lane], &out5[lane], &out6[lane], &out7[lane], &out8[lane], &out9[lane]);
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    pack_out[elements[0][element]] += out0[lane];
                    pack_out[elements[1][element]] += out1[lane];
                    pack_out[elements[2][element]] += out2[lane];
                    pack_out[elements[3][element]] += out3[lane];
                    pack_out[elements[4][element]] += out4[lane];
                    pack_out[elements[5][element]] += out5[lane];
                    pack_out[elements[6][element]] += out6[lane];
                    pack_out[elements[7][element]] += out7[lane];
                    pack_out[elements[8][element]] += out8[lane];
                    pack_out[elements[9][element]] += out9[lane];
                }
            }

            scalar_t *const SFEM_RESTRICT acc = &u_out[owned_nodes_ptr[pack]];
            for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                acc[k] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                acc[k] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                u_out[ghosts[k]] += ghost_out[k];
                ghost_out[k] = scalar_t(0);
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    if (kappa == scalar_t(1)) {
        return laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl_kernel<scalar_t, jacobian_t, true>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_geom_metric, kappa, direction_stride, u_direction, out_stride, u_out);
    }
    return laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl_kernel<scalar_t, jacobian_t, false>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_geom_metric, kappa, direction_stride, u_direction, out_stride, u_out);
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet10_private_metric_jacobian_action_packed_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl<double, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_geom_metric, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet10_private_metric_jacobian_action_packed_mesh_soa_impl<float, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_geom_metric, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, bool UnitKappa>
static SFEM_INLINE int laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl_kernel(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
    static constexpr int VECTOR_SIZE = 32;
    (void)nnodes;
    (void)max_nodes_per_pack;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_direction = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)max_nodes_per_pack);
        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];
        scalar_t out4[VECTOR_SIZE];
        scalar_t out5[VECTOR_SIZE];
        scalar_t out6[VECTOR_SIZE];
        scalar_t out7[VECTOR_SIZE];
        scalar_t out8[VECTOR_SIZE];
        scalar_t out9[VECTOR_SIZE];
        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];
        scalar_t u4[VECTOR_SIZE];
        scalar_t u5[VECTOR_SIZE];
        scalar_t u6[VECTOR_SIZE];
        scalar_t u7[VECTOR_SIZE];
        scalar_t u8[VECTOR_SIZE];
        scalar_t u9[VECTOR_SIZE];
        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                pack_direction[k] = u_direction[(owned_nodes_ptr[pack] + k) * direction_stride];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                pack_direction[n_contiguous + k] = u_direction[ghosts[k] * direction_stride];
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                const scalar_t metric_factor = UnitKappa ? scalar_t(1) : kappa;

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    const scalar_t inv_det = (metric_factor * scalar_t(1.0 / 6.0)) / scalar_t(g_jacobian_determinant0[element]);
                    const scalar_t adj0 = scalar_t(g_jacobian_adjugate0[element]);
                    const scalar_t adj1 = scalar_t(g_jacobian_adjugate1[element]);
                    const scalar_t adj2 = scalar_t(g_jacobian_adjugate2[element]);
                    const scalar_t adj3 = scalar_t(g_jacobian_adjugate3[element]);
                    const scalar_t adj4 = scalar_t(g_jacobian_adjugate4[element]);
                    const scalar_t adj5 = scalar_t(g_jacobian_adjugate5[element]);
                    const scalar_t adj6 = scalar_t(g_jacobian_adjugate6[element]);
                    const scalar_t adj7 = scalar_t(g_jacobian_adjugate7[element]);
                    const scalar_t adj8 = scalar_t(g_jacobian_adjugate8[element]);
                    fff0[lane] = (adj0 * adj0 + adj1 * adj1 + adj2 * adj2) * inv_det;
                    fff1[lane] = (adj0 * adj3 + adj1 * adj4 + adj2 * adj5) * inv_det;
                    fff2[lane] = (adj0 * adj6 + adj1 * adj7 + adj2 * adj8) * inv_det;
                    fff3[lane] = (adj3 * adj3 + adj4 * adj4 + adj5 * adj5) * inv_det;
                    fff4[lane] = (adj3 * adj6 + adj4 * adj7 + adj5 * adj8) * inv_det;
                    fff5[lane] = (adj6 * adj6 + adj7 * adj7 + adj8 * adj8) * inv_det;
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    u0[lane] = pack_direction[elements[0][element]];
                    u1[lane] = pack_direction[elements[1][element]];
                    u2[lane] = pack_direction[elements[2][element]];
                    u3[lane] = pack_direction[elements[3][element]];
                    u4[lane] = pack_direction[elements[4][element]];
                    u5[lane] = pack_direction[elements[5][element]];
                    u6[lane] = pack_direction[elements[6][element]];
                    u7[lane] = pack_direction[elements[7][element]];
                    u8[lane] = pack_direction[elements[8][element]];
                    u9[lane] = pack_direction[elements[9][element]];
                }

#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    tet10_laplacian_apply_fff_soa_tpl<scalar_t, scalar_t>(fff0[lane], fff1[lane], fff2[lane], fff3[lane], fff4[lane], fff5[lane], u0[lane], u1[lane], u2[lane], u3[lane], u4[lane], u5[lane], u6[lane], u7[lane], u8[lane], u9[lane], &out0[lane], &out1[lane], &out2[lane], &out3[lane], &out4[lane], &out5[lane], &out6[lane], &out7[lane], &out8[lane], &out9[lane]);
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    pack_out[elements[0][element]] += out0[lane];
                    pack_out[elements[1][element]] += out1[lane];
                    pack_out[elements[2][element]] += out2[lane];
                    pack_out[elements[3][element]] += out3[lane];
                    pack_out[elements[4][element]] += out4[lane];
                    pack_out[elements[5][element]] += out5[lane];
                    pack_out[elements[6][element]] += out6[lane];
                    pack_out[elements[7][element]] += out7[lane];
                    pack_out[elements[8][element]] += out8[lane];
                    pack_out[elements[9][element]] += out9[lane];
                }
            }

            for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                u_out[ghosts[k] * out_stride] += pack_out[n_contiguous + k];
                pack_out[n_contiguous + k] = scalar_t(0);
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
    if (kappa == scalar_t(1)) {
        return laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl_kernel<scalar_t, jacobian_t, true>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
    }
    return laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl_kernel<scalar_t, jacobian_t, false>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet10_jacobian_action_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
    return sfem::codegen::laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl<double, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_jacobian_action_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
    return sfem::codegen::laplace_tet10_jacobian_action_packed_affine_mesh_soa_impl<float, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet10_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 10; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction, kappa, block_output);

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

extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_tet10_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_tet10_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet10_jacobian_action_packed_isoparametric_mesh_soa_impl(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_direction = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    pack_coordinate[k] = scalar_t(coordinate_component[owned_nodes_ptr[pack] + k]);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[ghosts[k]]);
                }
            }
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const idx_t node = owned_nodes_ptr[pack] + k;
                pack_direction[k] = u_direction[node * direction_stride];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                pack_direction[n_contiguous + k] = u_direction[ghosts[k] * direction_stride];
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];
                scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
                scalar_t block_determinant[N_QP * VECTOR_SIZE];
                scalar_t block_direction[N_STREAMS][VECTOR_SIZE];
                scalar_t block_output[N_STREAMS][VECTOR_SIZE];

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT coordinate_shape = elements[shape];
                    const uint16_t *const SFEM_RESTRICT field_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            block_coordinates[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + coordinate_shape[evbegin + lane]];
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_direction[shape][lane] = pack_direction[field_shape[evbegin + lane]];
                        block_output[shape][lane] = scalar_t(0);
                    }
                }

                scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
                for (int q = 0; q < N_QP; ++q) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                        const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                        const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                        const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                        const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                        const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                        const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 9];
                        const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 9];
                        const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 9];
                        geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                                J00, J01, J02, J10, J11, J12, J20, J21, J22,
                                block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
                    }
                }
                const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

                laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction, kappa, block_output);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT field_shape = elements[shape];
                    for (int lane = 0; lane < nelems; ++lane) {
                        pack_out[field_shape[evbegin + lane]] += block_output[shape][lane];
                    }
                }
            }

            for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                u_out[ghosts[k] * out_stride] += pack_out[n_contiguous + k];
                pack_out[n_contiguous + k] = scalar_t(0);
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet10_jacobian_action_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet10_jacobian_action_packed_isoparametric_mesh_soa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_jacobian_action_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet10_jacobian_action_packed_isoparametric_mesh_soa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
