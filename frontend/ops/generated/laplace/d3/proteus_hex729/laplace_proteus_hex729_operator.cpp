#include <type_traits>
#include "../laplace_d3_tensor_product_local.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
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
struct laplace_proteus_hex729_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[81] = {scalar_t(0.69807186203642757), scalar_t(0.81505010973104985), scalar_t(-1.3293319092588429), scalar_t(1.7331517697232279), scalar_t(-1.6070179285100861), scalar_t(1.0217709047461547), scalar_t(-0.42389129488512961), scalar_t(0.10348949013079144), scalar_t(-0.011293003713592814), scalar_t(0.097272052747715651), scalar_t(1.4831463895078723), scalar_t(-1.3290095232907182), scalar_t(1.5241120690547567), scalar_t(-1.3354423592773774), scalar_t(0.82242311246193067), scalar_t(-0.33426507759436569), scalar_t(0.080450330152689833), scalar_t(-0.0086869937625030282), scalar_t(-0.011864930096761444), scalar_t(0.26860098236210328), scalar_t(1.1329572434965776), scalar_t(-0.70696248700185693), scalar_t(0.52352041127100157), scalar_t(-0.29754282112844066), scalar_t(0.11536579986803044), scalar_t(-0.02691751236319731), scalar_t(0.002843313592543222), scalar_t(0.0019716993816209865), scalar_t(-0.025035909725628615), scalar_t(0.21227346628503463), scalar_t(1.0048381149760741), scalar_t(-0.28763254532661015), scalar_t(0.12992986539174153), scalar_t(-0.045260758332968602), scalar_t(0.0099221958482425942), scalar_t(-0.0010061284975065718), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-0.0010061284975065728), scalar_t(0.0099221958482426029), scalar_t(-0.045260758332968637), scalar_t(0.12992986539174167), scalar_t(-0.28763254532661037), scalar_t(1.0048381149760741), scalar_t(0.21227346628503507), scalar_t(-0.025035909725628643), scalar_t(0.0019716993816209891), scalar_t(0.0028433135925432225), scalar_t(-0.026917512363197314), scalar_t(0.11536579986803044), scalar_t(-0.29754282112844055), scalar_t(0.52352041127100168), scalar_t(-0.70696248700185704), scalar_t(1.1329572434965776), scalar_t(0.26860098236210328), scalar_t(-0.011864930096761444), scalar_t(-0.0086869937625030282), scalar_t(0.080450330152689833), scalar_t(-0.33426507759436574), scalar_t(0.82242311246193045), scalar_t(-1.3354423592773779), scalar_t(1.5241120690547563), scalar_t(-1.3290095232907184), scalar_t(1.4831463895078723), scalar_t(0.097272052747715651), scalar_t(-0.011293003713592816), scalar_t(0.10348949013079145), scalar_t(-0.42389129488512955), scalar_t(1.0217709047461545), scalar_t(-1.6070179285100863), scalar_t(1.7331517697232279), scalar_t(-1.3293319092588425), scalar_t(0.81505010973104974), scalar_t(0.69807186203642768)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[81] = {scalar_t(-16.386933245281014), scalar_t(39.536085006311602), scalar_t(-57.974849142557389), scalar_t(73.008796001230877), scalar_t(-66.539785287489011), scalar_t(41.874008884964852), scalar_t(-17.253319732828214), scalar_t(4.1917397279744941), scalar_t(-0.45574221232625323), scalar_t(-3.9583045868241791), scalar_t(-7.7839860203350746), scalar_t(29.961014736629757), scalar_t(-38.229164957492692), scalar_t(34.859642229748758), scalar_t(-21.920972267544382), scalar_t(9.0247313732825045), scalar_t(-2.1910396583029774), scalar_t(0.23807915083829612), scalar_t(0.22053364437663764), scalar_t(-7.5348805835990564), scalar_t(4.7890510370393846), scalar_t(5.5921315283090234), scalar_t(-5.3155254349266352), scalar_t(3.3020089257774603), scalar_t(-1.3402906398380594), scalar_t(0.32158730844917438), scalar_t(-0.034615785587925343), scalar_t(-0.051868335251774145), scalar_t(0.70211587272463594), scalar_t(-7.3715664437642845), scalar_t(3.6054224779714437), scalar_t(4.9411552165284398), scalar_t(-2.5809189409395019), scalar_t(0.94686791130811188), scalar_t(-0.21317802024125845), scalar_t(0.021970261664185264), scalar_t(0.028571428571428571), scalar_t(-0.30476190476190473), scalar_t(1.5999999999999999), scalar_t(-6.4000000000000004), scalar_t(-2.2204460492503131e-15), scalar_t(6.4000000000000004), scalar_t(-1.5999999999999996), scalar_t(0.30476190476190479), scalar_t(-0.028571428571428567), scalar_t(-0.021970261664185239), scalar_t(0.21317802024125815), scalar_t(-0.94686791130811043), scalar_t(2.580918940939497), scalar_t(-4.9411552165284309), scalar_t(-3.6054224779714543), scalar_t(7.371566443764288), scalar_t(-0.7021158727246356), scalar_t(0.051868335251774103), scalar_t(0.034615785587925336), scalar_t(-0.32158730844917438), scalar_t(1.3402906398380607), scalar_t(-3.3020089257774625), scalar_t(5.3155254349266343), scalar_t(-5.5921315283090216), scalar_t(-4.7890510370393802), scalar_t(7.5348805835990564), scalar_t(-0.22053364437663761), scalar_t(-0.23807915083829601), scalar_t(2.1910396583029761), scalar_t(-9.0247313732825063), scalar_t(21.920972267544386), scalar_t(-34.859642229748758), scalar_t(38.229164957492685), scalar_t(-29.96101473662976), scalar_t(7.7839860203350746), scalar_t(3.95830458682418), scalar_t(0.45574221232625334), scalar_t(-4.191739727974495), scalar_t(17.25331973282821), scalar_t(-41.874008884964852), scalar_t(66.539785287488996), scalar_t(-73.008796001230905), scalar_t(57.974849142557375), scalar_t(-39.536085006311609), scalar_t(16.386933245281018)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[9] = {scalar_t(0.040637194180787095), scalar_t(0.09032408034742867), scalar_t(0.13030534820146775), scalar_t(0.15617353852000149), scalar_t(0.16511967750062995), scalar_t(0.15617353852000149), scalar_t(0.13030534820146775), scalar_t(0.09032408034742867), scalar_t(0.040637194180787095)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_proteus_hex729_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[81] = {scalar_t(0.69807186203642757), scalar_t(0.81505010973104985), scalar_t(-1.3293319092588429), scalar_t(1.7331517697232279), scalar_t(-1.6070179285100861), scalar_t(1.0217709047461547), scalar_t(-0.42389129488512961), scalar_t(0.10348949013079144), scalar_t(-0.011293003713592814), scalar_t(0.097272052747715651), scalar_t(1.4831463895078723), scalar_t(-1.3290095232907182), scalar_t(1.5241120690547567), scalar_t(-1.3354423592773774), scalar_t(0.82242311246193067), scalar_t(-0.33426507759436569), scalar_t(0.080450330152689833), scalar_t(-0.0086869937625030282), scalar_t(-0.011864930096761444), scalar_t(0.26860098236210328), scalar_t(1.1329572434965776), scalar_t(-0.70696248700185693), scalar_t(0.52352041127100157), scalar_t(-0.29754282112844066), scalar_t(0.11536579986803044), scalar_t(-0.02691751236319731), scalar_t(0.002843313592543222), scalar_t(0.0019716993816209865), scalar_t(-0.025035909725628615), scalar_t(0.21227346628503463), scalar_t(1.0048381149760741), scalar_t(-0.28763254532661015), scalar_t(0.12992986539174153), scalar_t(-0.045260758332968602), scalar_t(0.0099221958482425942), scalar_t(-0.0010061284975065718), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-0.0010061284975065728), scalar_t(0.0099221958482426029), scalar_t(-0.045260758332968637), scalar_t(0.12992986539174167), scalar_t(-0.28763254532661037), scalar_t(1.0048381149760741), scalar_t(0.21227346628503507), scalar_t(-0.025035909725628643), scalar_t(0.0019716993816209891), scalar_t(0.0028433135925432225), scalar_t(-0.026917512363197314), scalar_t(0.11536579986803044), scalar_t(-0.29754282112844055), scalar_t(0.52352041127100168), scalar_t(-0.70696248700185704), scalar_t(1.1329572434965776), scalar_t(0.26860098236210328), scalar_t(-0.011864930096761444), scalar_t(-0.0086869937625030282), scalar_t(0.080450330152689833), scalar_t(-0.33426507759436574), scalar_t(0.82242311246193045), scalar_t(-1.3354423592773779), scalar_t(1.5241120690547563), scalar_t(-1.3290095232907184), scalar_t(1.4831463895078723), scalar_t(0.097272052747715651), scalar_t(-0.011293003713592816), scalar_t(0.10348949013079145), scalar_t(-0.42389129488512955), scalar_t(1.0217709047461545), scalar_t(-1.6070179285100863), scalar_t(1.7331517697232279), scalar_t(-1.3293319092588425), scalar_t(0.81505010973104974), scalar_t(0.69807186203642768)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[81] = {scalar_t(-16.386933245281014), scalar_t(39.536085006311602), scalar_t(-57.974849142557389), scalar_t(73.008796001230877), scalar_t(-66.539785287489011), scalar_t(41.874008884964852), scalar_t(-17.253319732828214), scalar_t(4.1917397279744941), scalar_t(-0.45574221232625323), scalar_t(-3.9583045868241791), scalar_t(-7.7839860203350746), scalar_t(29.961014736629757), scalar_t(-38.229164957492692), scalar_t(34.859642229748758), scalar_t(-21.920972267544382), scalar_t(9.0247313732825045), scalar_t(-2.1910396583029774), scalar_t(0.23807915083829612), scalar_t(0.22053364437663764), scalar_t(-7.5348805835990564), scalar_t(4.7890510370393846), scalar_t(5.5921315283090234), scalar_t(-5.3155254349266352), scalar_t(3.3020089257774603), scalar_t(-1.3402906398380594), scalar_t(0.32158730844917438), scalar_t(-0.034615785587925343), scalar_t(-0.051868335251774145), scalar_t(0.70211587272463594), scalar_t(-7.3715664437642845), scalar_t(3.6054224779714437), scalar_t(4.9411552165284398), scalar_t(-2.5809189409395019), scalar_t(0.94686791130811188), scalar_t(-0.21317802024125845), scalar_t(0.021970261664185264), scalar_t(0.028571428571428571), scalar_t(-0.30476190476190473), scalar_t(1.5999999999999999), scalar_t(-6.4000000000000004), scalar_t(-2.2204460492503131e-15), scalar_t(6.4000000000000004), scalar_t(-1.5999999999999996), scalar_t(0.30476190476190479), scalar_t(-0.028571428571428567), scalar_t(-0.021970261664185239), scalar_t(0.21317802024125815), scalar_t(-0.94686791130811043), scalar_t(2.580918940939497), scalar_t(-4.9411552165284309), scalar_t(-3.6054224779714543), scalar_t(7.371566443764288), scalar_t(-0.7021158727246356), scalar_t(0.051868335251774103), scalar_t(0.034615785587925336), scalar_t(-0.32158730844917438), scalar_t(1.3402906398380607), scalar_t(-3.3020089257774625), scalar_t(5.3155254349266343), scalar_t(-5.5921315283090216), scalar_t(-4.7890510370393802), scalar_t(7.5348805835990564), scalar_t(-0.22053364437663761), scalar_t(-0.23807915083829601), scalar_t(2.1910396583029761), scalar_t(-9.0247313732825063), scalar_t(21.920972267544386), scalar_t(-34.859642229748758), scalar_t(38.229164957492685), scalar_t(-29.96101473662976), scalar_t(7.7839860203350746), scalar_t(3.95830458682418), scalar_t(0.45574221232625334), scalar_t(-4.191739727974495), scalar_t(17.25331973282821), scalar_t(-41.874008884964852), scalar_t(66.539785287488996), scalar_t(-73.008796001230905), scalar_t(57.974849142557375), scalar_t(-39.536085006311609), scalar_t(16.386933245281018)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[9] = {scalar_t(0.040637194180787095), scalar_t(0.09032408034742867), scalar_t(0.13030534820146775), scalar_t(0.15617353852000149), scalar_t(0.16511967750062995), scalar_t(0.15617353852000149), scalar_t(0.13030534820146775), scalar_t(0.09032408034742867), scalar_t(0.040637194180787095)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex729_residual_element_soa_diagnostics_data = {
    "laplace_proteus_hex729_residual_element_soa",
    "PROTEUS_HEX729",
    3,
    729,
    729,
    16,
    9,
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
    162,
    9,
    1,
    729,
    0,
    729,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_residual_element_soa",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_residual_element_soa_float",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_residual_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex729_jacobian_u_u_diagnostics_data = {
    "laplace_proteus_hex729_jacobian_u_u",
    "PROTEUS_HEX729",
    3,
    729,
    729,
    16,
    9,
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
    162,
    9,
    1,
    0,
    729,
    729,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex729_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_jacobian_u_u",
            &sfem::codegen::laplace_proteus_hex729_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_jacobian_u_u_float",
            &sfem::codegen::laplace_proteus_hex729_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data = {
    "laplace_proteus_hex729_jacobian_action_element_soa",
    "PROTEUS_HEX729",
    3,
    729,
    729,
    16,
    9,
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
    162,
    9,
    1,
    0,
    729,
    729,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_jacobian_action_element_soa",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_proteus_hex729_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[729],
        const double kappa,
        double *const SFEM_RESTRICT output[729]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<double, 729, 729, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[729],
        const float kappa,
        float *const SFEM_RESTRICT output[729]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<float, 729, 729, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex729_residual_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();

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

        for (int stream = 0; stream < 729; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t * block_current_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_current_streams[stream] = block_current[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
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
        const scalar_t *const block_adjugate[9] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current_streams, kappa, block_output_streams);

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

extern "C" int laplace_proteus_hex729_residual_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex729_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_residual_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex729_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex729_residual_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();

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

        for (int stream = 0; stream < 729; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191], block_coordinates[192], block_coordinates[193], block_coordinates[194], block_coordinates[195], block_coordinates[196], block_coordinates[197], block_coordinates[198], block_coordinates[199], block_coordinates[200], block_coordinates[201], block_coordinates[202], block_coordinates[203], block_coordinates[204], block_coordinates[205], block_coordinates[206], block_coordinates[207], block_coordinates[208], block_coordinates[209], block_coordinates[210], block_coordinates[211], block_coordinates[212], block_coordinates[213], block_coordinates[214], block_coordinates[215], block_coordinates[216], block_coordinates[217], block_coordinates[218], block_coordinates[219], block_coordinates[220], block_coordinates[221], block_coordinates[222], block_coordinates[223], block_coordinates[224], block_coordinates[225], block_coordinates[226], block_coordinates[227], block_coordinates[228], block_coordinates[229], block_coordinates[230], block_coordinates[231], block_coordinates[232], block_coordinates[233], block_coordinates[234], block_coordinates[235], block_coordinates[236], block_coordinates[237], block_coordinates[238], block_coordinates[239], block_coordinates[240], block_coordinates[241], block_coordinates[242], block_coordinates[243], block_coordinates[244], block_coordinates[245], block_coordinates[246], block_coordinates[247], block_coordinates[248], block_coordinates[249], block_coordinates[250], block_coordinates[251], block_coordinates[252], block_coordinates[253], block_coordinates[254], block_coordinates[255], block_coordinates[256], block_coordinates[257], block_coordinates[258], block_coordinates[259], block_coordinates[260], block_coordinates[261], block_coordinates[262], block_coordinates[263], block_coordinates[264], block_coordinates[265], block_coordinates[266], block_coordinates[267], block_coordinates[268], block_coordinates[269], block_coordinates[270], block_coordinates[271], block_coordinates[272], block_coordinates[273], block_coordinates[274], block_coordinates[275], block_coordinates[276], block_coordinates[277], block_coordinates[278], block_coordinates[279], block_coordinates[280], block_coordinates[281], block_coordinates[282], block_coordinates[283], block_coordinates[284], block_coordinates[285], block_coordinates[286], block_coordinates[287], block_coordinates[288], block_coordinates[289], block_coordinates[290], block_coordinates[291], block_coordinates[292], block_coordinates[293], block_coordinates[294], block_coordinates[295], block_coordinates[296], block_coordinates[297], block_coordinates[298], block_coordinates[299], block_coordinates[300], block_coordinates[301], block_coordinates[302], block_coordinates[303], block_coordinates[304], block_coordinates[305], block_coordinates[306], block_coordinates[307], block_coordinates[308], block_coordinates[309], block_coordinates[310], block_coordinates[311], block_coordinates[312], block_coordinates[313], block_coordinates[314], block_coordinates[315], block_coordinates[316], block_coordinates[317], block_coordinates[318], block_coordinates[319], block_coordinates[320], block_coordinates[321], block_coordinates[322], block_coordinates[323], block_coordinates[324], block_coordinates[325], block_coordinates[326], block_coordinates[327], block_coordinates[328], block_coordinates[329], block_coordinates[330], block_coordinates[331], block_coordinates[332], block_coordinates[333], block_coordinates[334], block_coordinates[335], block_coordinates[336], block_coordinates[337], block_coordinates[338], block_coordinates[339], block_coordinates[340], block_coordinates[341], block_coordinates[342], block_coordinates[343], block_coordinates[344], block_coordinates[345], block_coordinates[346], block_coordinates[347], block_coordinates[348], block_coordinates[349], block_coordinates[350], block_coordinates[351], block_coordinates[352], block_coordinates[353], block_coordinates[354], block_coordinates[355], block_coordinates[356], block_coordinates[357], block_coordinates[358], block_coordinates[359], block_coordinates[360], block_coordinates[361], block_coordinates[362], block_coordinates[363], block_coordinates[364], block_coordinates[365], block_coordinates[366], block_coordinates[367], block_coordinates[368], block_coordinates[369], block_coordinates[370], block_coordinates[371], block_coordinates[372], block_coordinates[373], block_coordinates[374], block_coordinates[375], block_coordinates[376], block_coordinates[377], block_coordinates[378], block_coordinates[379], block_coordinates[380], block_coordinates[381], block_coordinates[382], block_coordinates[383], block_coordinates[384], block_coordinates[385], block_coordinates[386], block_coordinates[387], block_coordinates[388], block_coordinates[389], block_coordinates[390], block_coordinates[391], block_coordinates[392], block_coordinates[393], block_coordinates[394], block_coordinates[395], block_coordinates[396], block_coordinates[397], block_coordinates[398], block_coordinates[399], block_coordinates[400], block_coordinates[401], block_coordinates[402], block_coordinates[403], block_coordinates[404], block_coordinates[405], block_coordinates[406], block_coordinates[407], block_coordinates[408], block_coordinates[409], block_coordinates[410], block_coordinates[411], block_coordinates[412], block_coordinates[413], block_coordinates[414], block_coordinates[415], block_coordinates[416], block_coordinates[417], block_coordinates[418], block_coordinates[419], block_coordinates[420], block_coordinates[421], block_coordinates[422], block_coordinates[423], block_coordinates[424], block_coordinates[425], block_coordinates[426], block_coordinates[427], block_coordinates[428], block_coordinates[429], block_coordinates[430], block_coordinates[431], block_coordinates[432], block_coordinates[433], block_coordinates[434], block_coordinates[435], block_coordinates[436], block_coordinates[437], block_coordinates[438], block_coordinates[439], block_coordinates[440], block_coordinates[441], block_coordinates[442], block_coordinates[443], block_coordinates[444], block_coordinates[445], block_coordinates[446], block_coordinates[447], block_coordinates[448], block_coordinates[449], block_coordinates[450], block_coordinates[451], block_coordinates[452], block_coordinates[453], block_coordinates[454], block_coordinates[455], block_coordinates[456], block_coordinates[457], block_coordinates[458], block_coordinates[459], block_coordinates[460], block_coordinates[461], block_coordinates[462], block_coordinates[463], block_coordinates[464], block_coordinates[465], block_coordinates[466], block_coordinates[467], block_coordinates[468], block_coordinates[469], block_coordinates[470], block_coordinates[471], block_coordinates[472], block_coordinates[473], block_coordinates[474], block_coordinates[475], block_coordinates[476], block_coordinates[477], block_coordinates[478], block_coordinates[479], block_coordinates[480], block_coordinates[481], block_coordinates[482], block_coordinates[483], block_coordinates[484], block_coordinates[485], block_coordinates[486], block_coordinates[487], block_coordinates[488], block_coordinates[489], block_coordinates[490], block_coordinates[491], block_coordinates[492], block_coordinates[493], block_coordinates[494], block_coordinates[495], block_coordinates[496], block_coordinates[497], block_coordinates[498], block_coordinates[499], block_coordinates[500], block_coordinates[501], block_coordinates[502], block_coordinates[503], block_coordinates[504], block_coordinates[505], block_coordinates[506], block_coordinates[507], block_coordinates[508], block_coordinates[509], block_coordinates[510], block_coordinates[511], block_coordinates[512], block_coordinates[513], block_coordinates[514], block_coordinates[515], block_coordinates[516], block_coordinates[517], block_coordinates[518], block_coordinates[519], block_coordinates[520], block_coordinates[521], block_coordinates[522], block_coordinates[523], block_coordinates[524], block_coordinates[525], block_coordinates[526], block_coordinates[527], block_coordinates[528], block_coordinates[529], block_coordinates[530], block_coordinates[531], block_coordinates[532], block_coordinates[533], block_coordinates[534], block_coordinates[535], block_coordinates[536], block_coordinates[537], block_coordinates[538], block_coordinates[539], block_coordinates[540], block_coordinates[541], block_coordinates[542], block_coordinates[543], block_coordinates[544], block_coordinates[545], block_coordinates[546], block_coordinates[547], block_coordinates[548], block_coordinates[549], block_coordinates[550], block_coordinates[551], block_coordinates[552], block_coordinates[553], block_coordinates[554], block_coordinates[555], block_coordinates[556], block_coordinates[557], block_coordinates[558], block_coordinates[559], block_coordinates[560], block_coordinates[561], block_coordinates[562], block_coordinates[563], block_coordinates[564], block_coordinates[565], block_coordinates[566], block_coordinates[567], block_coordinates[568], block_coordinates[569], block_coordinates[570], block_coordinates[571], block_coordinates[572], block_coordinates[573], block_coordinates[574], block_coordinates[575], block_coordinates[576], block_coordinates[577], block_coordinates[578], block_coordinates[579], block_coordinates[580], block_coordinates[581], block_coordinates[582], block_coordinates[583], block_coordinates[584], block_coordinates[585], block_coordinates[586], block_coordinates[587], block_coordinates[588], block_coordinates[589], block_coordinates[590], block_coordinates[591], block_coordinates[592], block_coordinates[593], block_coordinates[594], block_coordinates[595], block_coordinates[596], block_coordinates[597], block_coordinates[598], block_coordinates[599], block_coordinates[600], block_coordinates[601], block_coordinates[602], block_coordinates[603], block_coordinates[604], block_coordinates[605], block_coordinates[606], block_coordinates[607], block_coordinates[608], block_coordinates[609], block_coordinates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613], block_coordinates[614], block_coordinates[615], block_coordinates[616], block_coordinates[617], block_coordinates[618], block_coordinates[619], block_coordinates[620], block_coordinates[621], block_coordinates[622], block_coordinates[623], block_coordinates[624], block_coordinates[625], block_coordinates[626], block_coordinates[627], block_coordinates[628], block_coordinates[629], block_coordinates[630], block_coordinates[631], block_coordinates[632], block_coordinates[633], block_coordinates[634], block_coordinates[635], block_coordinates[636], block_coordinates[637], block_coordinates[638], block_coordinates[639], block_coordinates[640], block_coordinates[641], block_coordinates[642], block_coordinates[643], block_coordinates[644], block_coordinates[645], block_coordinates[646], block_coordinates[647], block_coordinates[648], block_coordinates[649], block_coordinates[650], block_coordinates[651], block_coordinates[652], block_coordinates[653], block_coordinates[654], block_coordinates[655], block_coordinates[656], block_coordinates[657], block_coordinates[658], block_coordinates[659], block_coordinates[660], block_coordinates[661], block_coordinates[662], block_coordinates[663], block_coordinates[664], block_coordinates[665], block_coordinates[666], block_coordinates[667], block_coordinates[668], block_coordinates[669], block_coordinates[670], block_coordinates[671], block_coordinates[672], block_coordinates[673], block_coordinates[674], block_coordinates[675], block_coordinates[676], block_coordinates[677], block_coordinates[678], block_coordinates[679], block_coordinates[680], block_coordinates[681], block_coordinates[682], block_coordinates[683], block_coordinates[684], block_coordinates[685], block_coordinates[686], block_coordinates[687], block_coordinates[688], block_coordinates[689], block_coordinates[690], block_coordinates[691], block_coordinates[692], block_coordinates[693], block_coordinates[694], block_coordinates[695], block_coordinates[696], block_coordinates[697], block_coordinates[698], block_coordinates[699], block_coordinates[700], block_coordinates[701], block_coordinates[702], block_coordinates[703], block_coordinates[704], block_coordinates[705], block_coordinates[706], block_coordinates[707], block_coordinates[708], block_coordinates[709], block_coordinates[710], block_coordinates[711], block_coordinates[712], block_coordinates[713], block_coordinates[714], block_coordinates[715], block_coordinates[716], block_coordinates[717], block_coordinates[718], block_coordinates[719], block_coordinates[720], block_coordinates[721], block_coordinates[722], block_coordinates[723], block_coordinates[724], block_coordinates[725], block_coordinates[726], block_coordinates[727], block_coordinates[728], block_coordinates[729], block_coordinates[730], block_coordinates[731], block_coordinates[732], block_coordinates[733], block_coordinates[734], block_coordinates[735], block_coordinates[736], block_coordinates[737], block_coordinates[738], block_coordinates[739], block_coordinates[740], block_coordinates[741], block_coordinates[742], block_coordinates[743], block_coordinates[744], block_coordinates[745], block_coordinates[746], block_coordinates[747], block_coordinates[748], block_coordinates[749], block_coordinates[750], block_coordinates[751], block_coordinates[752], block_coordinates[753], block_coordinates[754], block_coordinates[755], block_coordinates[756], block_coordinates[757], block_coordinates[758], block_coordinates[759], block_coordinates[760], block_coordinates[761], block_coordinates[762], block_coordinates[763], block_coordinates[764], block_coordinates[765], block_coordinates[766], block_coordinates[767], block_coordinates[768], block_coordinates[769], block_coordinates[770], block_coordinates[771], block_coordinates[772], block_coordinates[773], block_coordinates[774], block_coordinates[775], block_coordinates[776], block_coordinates[777], block_coordinates[778], block_coordinates[779], block_coordinates[780], block_coordinates[781], block_coordinates[782], block_coordinates[783], block_coordinates[784], block_coordinates[785], block_coordinates[786], block_coordinates[787], block_coordinates[788], block_coordinates[789], block_coordinates[790], block_coordinates[791], block_coordinates[792], block_coordinates[793], block_coordinates[794], block_coordinates[795], block_coordinates[796], block_coordinates[797], block_coordinates[798], block_coordinates[799], block_coordinates[800], block_coordinates[801], block_coordinates[802], block_coordinates[803], block_coordinates[804], block_coordinates[805], block_coordinates[806], block_coordinates[807], block_coordinates[808], block_coordinates[809], block_coordinates[810], block_coordinates[811], block_coordinates[812], block_coordinates[813], block_coordinates[814], block_coordinates[815], block_coordinates[816], block_coordinates[817], block_coordinates[818], block_coordinates[819], block_coordinates[820], block_coordinates[821], block_coordinates[822], block_coordinates[823], block_coordinates[824], block_coordinates[825], block_coordinates[826], block_coordinates[827], block_coordinates[828], block_coordinates[829], block_coordinates[830], block_coordinates[831], block_coordinates[832], block_coordinates[833], block_coordinates[834], block_coordinates[835], block_coordinates[836], block_coordinates[837], block_coordinates[838], block_coordinates[839], block_coordinates[840], block_coordinates[841], block_coordinates[842], block_coordinates[843], block_coordinates[844], block_coordinates[845], block_coordinates[846], block_coordinates[847], block_coordinates[848], block_coordinates[849], block_coordinates[850], block_coordinates[851], block_coordinates[852], block_coordinates[853], block_coordinates[854], block_coordinates[855], block_coordinates[856], block_coordinates[857], block_coordinates[858], block_coordinates[859], block_coordinates[860], block_coordinates[861], block_coordinates[862], block_coordinates[863], block_coordinates[864], block_coordinates[865], block_coordinates[866], block_coordinates[867], block_coordinates[868], block_coordinates[869], block_coordinates[870], block_coordinates[871], block_coordinates[872], block_coordinates[873], block_coordinates[874], block_coordinates[875], block_coordinates[876], block_coordinates[877], block_coordinates[878], block_coordinates[879], block_coordinates[880], block_coordinates[881], block_coordinates[882], block_coordinates[883], block_coordinates[884], block_coordinates[885], block_coordinates[886], block_coordinates[887], block_coordinates[888], block_coordinates[889], block_coordinates[890], block_coordinates[891], block_coordinates[892], block_coordinates[893], block_coordinates[894], block_coordinates[895], block_coordinates[896], block_coordinates[897], block_coordinates[898], block_coordinates[899], block_coordinates[900], block_coordinates[901], block_coordinates[902], block_coordinates[903], block_coordinates[904], block_coordinates[905], block_coordinates[906], block_coordinates[907], block_coordinates[908], block_coordinates[909], block_coordinates[910], block_coordinates[911], block_coordinates[912], block_coordinates[913], block_coordinates[914], block_coordinates[915], block_coordinates[916], block_coordinates[917], block_coordinates[918], block_coordinates[919], block_coordinates[920], block_coordinates[921], block_coordinates[922], block_coordinates[923], block_coordinates[924], block_coordinates[925], block_coordinates[926], block_coordinates[927], block_coordinates[928], block_coordinates[929], block_coordinates[930], block_coordinates[931], block_coordinates[932], block_coordinates[933], block_coordinates[934], block_coordinates[935], block_coordinates[936], block_coordinates[937], block_coordinates[938], block_coordinates[939], block_coordinates[940], block_coordinates[941], block_coordinates[942], block_coordinates[943], block_coordinates[944], block_coordinates[945], block_coordinates[946], block_coordinates[947], block_coordinates[948], block_coordinates[949], block_coordinates[950], block_coordinates[951], block_coordinates[952], block_coordinates[953], block_coordinates[954], block_coordinates[955], block_coordinates[956], block_coordinates[957], block_coordinates[958], block_coordinates[959], block_coordinates[960], block_coordinates[961], block_coordinates[962], block_coordinates[963], block_coordinates[964], block_coordinates[965], block_coordinates[966], block_coordinates[967], block_coordinates[968], block_coordinates[969], block_coordinates[970], block_coordinates[971], block_coordinates[972], block_coordinates[973], block_coordinates[974], block_coordinates[975], block_coordinates[976], block_coordinates[977], block_coordinates[978], block_coordinates[979], block_coordinates[980], block_coordinates[981], block_coordinates[982], block_coordinates[983], block_coordinates[984], block_coordinates[985], block_coordinates[986], block_coordinates[987], block_coordinates[988], block_coordinates[989], block_coordinates[990], block_coordinates[991], block_coordinates[992], block_coordinates[993], block_coordinates[994], block_coordinates[995], block_coordinates[996], block_coordinates[997], block_coordinates[998], block_coordinates[999], block_coordinates[1000], block_coordinates[1001], block_coordinates[1002], block_coordinates[1003], block_coordinates[1004], block_coordinates[1005], block_coordinates[1006], block_coordinates[1007], block_coordinates[1008], block_coordinates[1009], block_coordinates[1010], block_coordinates[1011], block_coordinates[1012], block_coordinates[1013], block_coordinates[1014], block_coordinates[1015], block_coordinates[1016], block_coordinates[1017], block_coordinates[1018], block_coordinates[1019], block_coordinates[1020], block_coordinates[1021], block_coordinates[1022], block_coordinates[1023], block_coordinates[1024], block_coordinates[1025], block_coordinates[1026], block_coordinates[1027], block_coordinates[1028], block_coordinates[1029], block_coordinates[1030], block_coordinates[1031], block_coordinates[1032], block_coordinates[1033], block_coordinates[1034], block_coordinates[1035], block_coordinates[1036], block_coordinates[1037], block_coordinates[1038], block_coordinates[1039], block_coordinates[1040], block_coordinates[1041], block_coordinates[1042], block_coordinates[1043], block_coordinates[1044], block_coordinates[1045], block_coordinates[1046], block_coordinates[1047], block_coordinates[1048], block_coordinates[1049], block_coordinates[1050], block_coordinates[1051], block_coordinates[1052], block_coordinates[1053], block_coordinates[1054], block_coordinates[1055], block_coordinates[1056], block_coordinates[1057], block_coordinates[1058], block_coordinates[1059], block_coordinates[1060], block_coordinates[1061], block_coordinates[1062], block_coordinates[1063], block_coordinates[1064], block_coordinates[1065], block_coordinates[1066], block_coordinates[1067], block_coordinates[1068], block_coordinates[1069], block_coordinates[1070], block_coordinates[1071], block_coordinates[1072], block_coordinates[1073], block_coordinates[1074], block_coordinates[1075], block_coordinates[1076], block_coordinates[1077], block_coordinates[1078], block_coordinates[1079], block_coordinates[1080], block_coordinates[1081], block_coordinates[1082], block_coordinates[1083], block_coordinates[1084], block_coordinates[1085], block_coordinates[1086], block_coordinates[1087], block_coordinates[1088], block_coordinates[1089], block_coordinates[1090], block_coordinates[1091], block_coordinates[1092], block_coordinates[1093], block_coordinates[1094], block_coordinates[1095], block_coordinates[1096], block_coordinates[1097], block_coordinates[1098], block_coordinates[1099], block_coordinates[1100], block_coordinates[1101], block_coordinates[1102], block_coordinates[1103], block_coordinates[1104], block_coordinates[1105], block_coordinates[1106], block_coordinates[1107], block_coordinates[1108], block_coordinates[1109], block_coordinates[1110], block_coordinates[1111], block_coordinates[1112], block_coordinates[1113], block_coordinates[1114], block_coordinates[1115], block_coordinates[1116], block_coordinates[1117], block_coordinates[1118], block_coordinates[1119], block_coordinates[1120], block_coordinates[1121], block_coordinates[1122], block_coordinates[1123], block_coordinates[1124], block_coordinates[1125], block_coordinates[1126], block_coordinates[1127], block_coordinates[1128], block_coordinates[1129], block_coordinates[1130], block_coordinates[1131], block_coordinates[1132], block_coordinates[1133], block_coordinates[1134], block_coordinates[1135], block_coordinates[1136], block_coordinates[1137], block_coordinates[1138], block_coordinates[1139], block_coordinates[1140], block_coordinates[1141], block_coordinates[1142], block_coordinates[1143], block_coordinates[1144], block_coordinates[1145], block_coordinates[1146], block_coordinates[1147], block_coordinates[1148], block_coordinates[1149], block_coordinates[1150], block_coordinates[1151], block_coordinates[1152], block_coordinates[1153], block_coordinates[1154], block_coordinates[1155], block_coordinates[1156], block_coordinates[1157], block_coordinates[1158], block_coordinates[1159], block_coordinates[1160], block_coordinates[1161], block_coordinates[1162], block_coordinates[1163], block_coordinates[1164], block_coordinates[1165], block_coordinates[1166], block_coordinates[1167], block_coordinates[1168], block_coordinates[1169], block_coordinates[1170], block_coordinates[1171], block_coordinates[1172], block_coordinates[1173], block_coordinates[1174], block_coordinates[1175], block_coordinates[1176], block_coordinates[1177], block_coordinates[1178], block_coordinates[1179], block_coordinates[1180], block_coordinates[1181], block_coordinates[1182], block_coordinates[1183], block_coordinates[1184], block_coordinates[1185], block_coordinates[1186], block_coordinates[1187], block_coordinates[1188], block_coordinates[1189], block_coordinates[1190], block_coordinates[1191], block_coordinates[1192], block_coordinates[1193], block_coordinates[1194], block_coordinates[1195], block_coordinates[1196], block_coordinates[1197], block_coordinates[1198], block_coordinates[1199], block_coordinates[1200], block_coordinates[1201], block_coordinates[1202], block_coordinates[1203], block_coordinates[1204], block_coordinates[1205], block_coordinates[1206], block_coordinates[1207], block_coordinates[1208], block_coordinates[1209], block_coordinates[1210], block_coordinates[1211], block_coordinates[1212], block_coordinates[1213], block_coordinates[1214], block_coordinates[1215], block_coordinates[1216], block_coordinates[1217], block_coordinates[1218], block_coordinates[1219], block_coordinates[1220], block_coordinates[1221], block_coordinates[1222], block_coordinates[1223], block_coordinates[1224], block_coordinates[1225], block_coordinates[1226], block_coordinates[1227], block_coordinates[1228], block_coordinates[1229], block_coordinates[1230], block_coordinates[1231], block_coordinates[1232], block_coordinates[1233], block_coordinates[1234], block_coordinates[1235], block_coordinates[1236], block_coordinates[1237], block_coordinates[1238], block_coordinates[1239], block_coordinates[1240], block_coordinates[1241], block_coordinates[1242], block_coordinates[1243], block_coordinates[1244], block_coordinates[1245], block_coordinates[1246], block_coordinates[1247], block_coordinates[1248], block_coordinates[1249], block_coordinates[1250], block_coordinates[1251], block_coordinates[1252], block_coordinates[1253], block_coordinates[1254], block_coordinates[1255], block_coordinates[1256], block_coordinates[1257], block_coordinates[1258], block_coordinates[1259], block_coordinates[1260], block_coordinates[1261], block_coordinates[1262], block_coordinates[1263], block_coordinates[1264], block_coordinates[1265], block_coordinates[1266], block_coordinates[1267], block_coordinates[1268], block_coordinates[1269], block_coordinates[1270], block_coordinates[1271], block_coordinates[1272], block_coordinates[1273], block_coordinates[1274], block_coordinates[1275], block_coordinates[1276], block_coordinates[1277], block_coordinates[1278], block_coordinates[1279], block_coordinates[1280], block_coordinates[1281], block_coordinates[1282], block_coordinates[1283], block_coordinates[1284], block_coordinates[1285], block_coordinates[1286], block_coordinates[1287], block_coordinates[1288], block_coordinates[1289], block_coordinates[1290], block_coordinates[1291], block_coordinates[1292], block_coordinates[1293], block_coordinates[1294], block_coordinates[1295], block_coordinates[1296], block_coordinates[1297], block_coordinates[1298], block_coordinates[1299], block_coordinates[1300], block_coordinates[1301], block_coordinates[1302], block_coordinates[1303], block_coordinates[1304], block_coordinates[1305], block_coordinates[1306], block_coordinates[1307], block_coordinates[1308], block_coordinates[1309], block_coordinates[1310], block_coordinates[1311], block_coordinates[1312], block_coordinates[1313], block_coordinates[1314], block_coordinates[1315], block_coordinates[1316], block_coordinates[1317], block_coordinates[1318], block_coordinates[1319], block_coordinates[1320], block_coordinates[1321], block_coordinates[1322], block_coordinates[1323], block_coordinates[1324], block_coordinates[1325], block_coordinates[1326], block_coordinates[1327], block_coordinates[1328], block_coordinates[1329], block_coordinates[1330], block_coordinates[1331], block_coordinates[1332], block_coordinates[1333], block_coordinates[1334], block_coordinates[1335], block_coordinates[1336], block_coordinates[1337], block_coordinates[1338], block_coordinates[1339], block_coordinates[1340], block_coordinates[1341], block_coordinates[1342], block_coordinates[1343], block_coordinates[1344], block_coordinates[1345], block_coordinates[1346], block_coordinates[1347], block_coordinates[1348], block_coordinates[1349], block_coordinates[1350], block_coordinates[1351], block_coordinates[1352], block_coordinates[1353], block_coordinates[1354], block_coordinates[1355], block_coordinates[1356], block_coordinates[1357], block_coordinates[1358], block_coordinates[1359], block_coordinates[1360], block_coordinates[1361], block_coordinates[1362], block_coordinates[1363], block_coordinates[1364], block_coordinates[1365], block_coordinates[1366], block_coordinates[1367], block_coordinates[1368], block_coordinates[1369], block_coordinates[1370], block_coordinates[1371], block_coordinates[1372], block_coordinates[1373], block_coordinates[1374], block_coordinates[1375], block_coordinates[1376], block_coordinates[1377], block_coordinates[1378], block_coordinates[1379], block_coordinates[1380], block_coordinates[1381], block_coordinates[1382], block_coordinates[1383], block_coordinates[1384], block_coordinates[1385], block_coordinates[1386], block_coordinates[1387], block_coordinates[1388], block_coordinates[1389], block_coordinates[1390], block_coordinates[1391], block_coordinates[1392], block_coordinates[1393], block_coordinates[1394], block_coordinates[1395], block_coordinates[1396], block_coordinates[1397], block_coordinates[1398], block_coordinates[1399], block_coordinates[1400], block_coordinates[1401], block_coordinates[1402], block_coordinates[1403], block_coordinates[1404], block_coordinates[1405], block_coordinates[1406], block_coordinates[1407], block_coordinates[1408], block_coordinates[1409], block_coordinates[1410], block_coordinates[1411], block_coordinates[1412], block_coordinates[1413], block_coordinates[1414], block_coordinates[1415], block_coordinates[1416], block_coordinates[1417], block_coordinates[1418], block_coordinates[1419], block_coordinates[1420], block_coordinates[1421], block_coordinates[1422], block_coordinates[1423], block_coordinates[1424], block_coordinates[1425], block_coordinates[1426], block_coordinates[1427], block_coordinates[1428], block_coordinates[1429], block_coordinates[1430], block_coordinates[1431], block_coordinates[1432], block_coordinates[1433], block_coordinates[1434], block_coordinates[1435], block_coordinates[1436], block_coordinates[1437], block_coordinates[1438], block_coordinates[1439], block_coordinates[1440], block_coordinates[1441], block_coordinates[1442], block_coordinates[1443], block_coordinates[1444], block_coordinates[1445], block_coordinates[1446], block_coordinates[1447], block_coordinates[1448], block_coordinates[1449], block_coordinates[1450], block_coordinates[1451], block_coordinates[1452], block_coordinates[1453], block_coordinates[1454], block_coordinates[1455], block_coordinates[1456], block_coordinates[1457], block_coordinates[1458], block_coordinates[1459], block_coordinates[1460], block_coordinates[1461], block_coordinates[1462], block_coordinates[1463], block_coordinates[1464], block_coordinates[1465], block_coordinates[1466], block_coordinates[1467], block_coordinates[1468], block_coordinates[1469], block_coordinates[1470], block_coordinates[1471], block_coordinates[1472], block_coordinates[1473], block_coordinates[1474], block_coordinates[1475], block_coordinates[1476], block_coordinates[1477], block_coordinates[1478], block_coordinates[1479], block_coordinates[1480], block_coordinates[1481], block_coordinates[1482], block_coordinates[1483], block_coordinates[1484], block_coordinates[1485], block_coordinates[1486], block_coordinates[1487], block_coordinates[1488], block_coordinates[1489], block_coordinates[1490], block_coordinates[1491], block_coordinates[1492], block_coordinates[1493], block_coordinates[1494], block_coordinates[1495], block_coordinates[1496], block_coordinates[1497], block_coordinates[1498], block_coordinates[1499], block_coordinates[1500], block_coordinates[1501], block_coordinates[1502], block_coordinates[1503], block_coordinates[1504], block_coordinates[1505], block_coordinates[1506], block_coordinates[1507], block_coordinates[1508], block_coordinates[1509], block_coordinates[1510], block_coordinates[1511], block_coordinates[1512], block_coordinates[1513], block_coordinates[1514], block_coordinates[1515], block_coordinates[1516], block_coordinates[1517], block_coordinates[1518], block_coordinates[1519], block_coordinates[1520], block_coordinates[1521], block_coordinates[1522], block_coordinates[1523], block_coordinates[1524], block_coordinates[1525], block_coordinates[1526], block_coordinates[1527], block_coordinates[1528], block_coordinates[1529], block_coordinates[1530], block_coordinates[1531], block_coordinates[1532], block_coordinates[1533], block_coordinates[1534], block_coordinates[1535], block_coordinates[1536], block_coordinates[1537], block_coordinates[1538], block_coordinates[1539], block_coordinates[1540], block_coordinates[1541], block_coordinates[1542], block_coordinates[1543], block_coordinates[1544], block_coordinates[1545], block_coordinates[1546], block_coordinates[1547], block_coordinates[1548], block_coordinates[1549], block_coordinates[1550], block_coordinates[1551], block_coordinates[1552], block_coordinates[1553], block_coordinates[1554], block_coordinates[1555], block_coordinates[1556], block_coordinates[1557], block_coordinates[1558], block_coordinates[1559], block_coordinates[1560], block_coordinates[1561], block_coordinates[1562], block_coordinates[1563], block_coordinates[1564], block_coordinates[1565], block_coordinates[1566], block_coordinates[1567], block_coordinates[1568], block_coordinates[1569], block_coordinates[1570], block_coordinates[1571], block_coordinates[1572], block_coordinates[1573], block_coordinates[1574], block_coordinates[1575], block_coordinates[1576], block_coordinates[1577], block_coordinates[1578], block_coordinates[1579], block_coordinates[1580], block_coordinates[1581], block_coordinates[1582], block_coordinates[1583], block_coordinates[1584], block_coordinates[1585], block_coordinates[1586], block_coordinates[1587], block_coordinates[1588], block_coordinates[1589], block_coordinates[1590], block_coordinates[1591], block_coordinates[1592], block_coordinates[1593], block_coordinates[1594], block_coordinates[1595], block_coordinates[1596], block_coordinates[1597], block_coordinates[1598], block_coordinates[1599], block_coordinates[1600], block_coordinates[1601], block_coordinates[1602], block_coordinates[1603], block_coordinates[1604], block_coordinates[1605], block_coordinates[1606], block_coordinates[1607], block_coordinates[1608], block_coordinates[1609], block_coordinates[1610], block_coordinates[1611], block_coordinates[1612], block_coordinates[1613], block_coordinates[1614], block_coordinates[1615], block_coordinates[1616], block_coordinates[1617], block_coordinates[1618], block_coordinates[1619], block_coordinates[1620], block_coordinates[1621], block_coordinates[1622], block_coordinates[1623], block_coordinates[1624], block_coordinates[1625], block_coordinates[1626], block_coordinates[1627], block_coordinates[1628], block_coordinates[1629], block_coordinates[1630], block_coordinates[1631], block_coordinates[1632], block_coordinates[1633], block_coordinates[1634], block_coordinates[1635], block_coordinates[1636], block_coordinates[1637], block_coordinates[1638], block_coordinates[1639], block_coordinates[1640], block_coordinates[1641], block_coordinates[1642], block_coordinates[1643], block_coordinates[1644], block_coordinates[1645], block_coordinates[1646], block_coordinates[1647], block_coordinates[1648], block_coordinates[1649], block_coordinates[1650], block_coordinates[1651], block_coordinates[1652], block_coordinates[1653], block_coordinates[1654], block_coordinates[1655], block_coordinates[1656], block_coordinates[1657], block_coordinates[1658], block_coordinates[1659], block_coordinates[1660], block_coordinates[1661], block_coordinates[1662], block_coordinates[1663], block_coordinates[1664], block_coordinates[1665], block_coordinates[1666], block_coordinates[1667], block_coordinates[1668], block_coordinates[1669], block_coordinates[1670], block_coordinates[1671], block_coordinates[1672], block_coordinates[1673], block_coordinates[1674], block_coordinates[1675], block_coordinates[1676], block_coordinates[1677], block_coordinates[1678], block_coordinates[1679], block_coordinates[1680], block_coordinates[1681], block_coordinates[1682], block_coordinates[1683], block_coordinates[1684], block_coordinates[1685], block_coordinates[1686], block_coordinates[1687], block_coordinates[1688], block_coordinates[1689], block_coordinates[1690], block_coordinates[1691], block_coordinates[1692], block_coordinates[1693], block_coordinates[1694], block_coordinates[1695], block_coordinates[1696], block_coordinates[1697], block_coordinates[1698], block_coordinates[1699], block_coordinates[1700], block_coordinates[1701], block_coordinates[1702], block_coordinates[1703], block_coordinates[1704], block_coordinates[1705], block_coordinates[1706], block_coordinates[1707], block_coordinates[1708], block_coordinates[1709], block_coordinates[1710], block_coordinates[1711], block_coordinates[1712], block_coordinates[1713], block_coordinates[1714], block_coordinates[1715], block_coordinates[1716], block_coordinates[1717], block_coordinates[1718], block_coordinates[1719], block_coordinates[1720], block_coordinates[1721], block_coordinates[1722], block_coordinates[1723], block_coordinates[1724], block_coordinates[1725], block_coordinates[1726], block_coordinates[1727], block_coordinates[1728], block_coordinates[1729], block_coordinates[1730], block_coordinates[1731], block_coordinates[1732], block_coordinates[1733], block_coordinates[1734], block_coordinates[1735], block_coordinates[1736], block_coordinates[1737], block_coordinates[1738], block_coordinates[1739], block_coordinates[1740], block_coordinates[1741], block_coordinates[1742], block_coordinates[1743], block_coordinates[1744], block_coordinates[1745], block_coordinates[1746], block_coordinates[1747], block_coordinates[1748], block_coordinates[1749], block_coordinates[1750], block_coordinates[1751], block_coordinates[1752], block_coordinates[1753], block_coordinates[1754], block_coordinates[1755], block_coordinates[1756], block_coordinates[1757], block_coordinates[1758], block_coordinates[1759], block_coordinates[1760], block_coordinates[1761], block_coordinates[1762], block_coordinates[1763], block_coordinates[1764], block_coordinates[1765], block_coordinates[1766], block_coordinates[1767], block_coordinates[1768], block_coordinates[1769], block_coordinates[1770], block_coordinates[1771], block_coordinates[1772], block_coordinates[1773], block_coordinates[1774], block_coordinates[1775], block_coordinates[1776], block_coordinates[1777], block_coordinates[1778], block_coordinates[1779], block_coordinates[1780], block_coordinates[1781], block_coordinates[1782], block_coordinates[1783], block_coordinates[1784], block_coordinates[1785], block_coordinates[1786], block_coordinates[1787], block_coordinates[1788], block_coordinates[1789], block_coordinates[1790], block_coordinates[1791], block_coordinates[1792], block_coordinates[1793], block_coordinates[1794], block_coordinates[1795], block_coordinates[1796], block_coordinates[1797], block_coordinates[1798], block_coordinates[1799], block_coordinates[1800], block_coordinates[1801], block_coordinates[1802], block_coordinates[1803], block_coordinates[1804], block_coordinates[1805], block_coordinates[1806], block_coordinates[1807], block_coordinates[1808], block_coordinates[1809], block_coordinates[1810], block_coordinates[1811], block_coordinates[1812], block_coordinates[1813], block_coordinates[1814], block_coordinates[1815], block_coordinates[1816], block_coordinates[1817], block_coordinates[1818], block_coordinates[1819], block_coordinates[1820], block_coordinates[1821], block_coordinates[1822], block_coordinates[1823], block_coordinates[1824], block_coordinates[1825], block_coordinates[1826], block_coordinates[1827], block_coordinates[1828], block_coordinates[1829], block_coordinates[1830], block_coordinates[1831], block_coordinates[1832], block_coordinates[1833], block_coordinates[1834], block_coordinates[1835], block_coordinates[1836], block_coordinates[1837], block_coordinates[1838], block_coordinates[1839], block_coordinates[1840], block_coordinates[1841], block_coordinates[1842], block_coordinates[1843], block_coordinates[1844], block_coordinates[1845], block_coordinates[1846], block_coordinates[1847], block_coordinates[1848], block_coordinates[1849], block_coordinates[1850], block_coordinates[1851], block_coordinates[1852], block_coordinates[1853], block_coordinates[1854], block_coordinates[1855], block_coordinates[1856], block_coordinates[1857], block_coordinates[1858], block_coordinates[1859], block_coordinates[1860], block_coordinates[1861], block_coordinates[1862], block_coordinates[1863], block_coordinates[1864], block_coordinates[1865], block_coordinates[1866], block_coordinates[1867], block_coordinates[1868], block_coordinates[1869], block_coordinates[1870], block_coordinates[1871], block_coordinates[1872], block_coordinates[1873], block_coordinates[1874], block_coordinates[1875], block_coordinates[1876], block_coordinates[1877], block_coordinates[1878], block_coordinates[1879], block_coordinates[1880], block_coordinates[1881], block_coordinates[1882], block_coordinates[1883], block_coordinates[1884], block_coordinates[1885], block_coordinates[1886], block_coordinates[1887], block_coordinates[1888], block_coordinates[1889], block_coordinates[1890], block_coordinates[1891], block_coordinates[1892], block_coordinates[1893], block_coordinates[1894], block_coordinates[1895], block_coordinates[1896], block_coordinates[1897], block_coordinates[1898], block_coordinates[1899], block_coordinates[1900], block_coordinates[1901], block_coordinates[1902], block_coordinates[1903], block_coordinates[1904], block_coordinates[1905], block_coordinates[1906], block_coordinates[1907], block_coordinates[1908], block_coordinates[1909], block_coordinates[1910], block_coordinates[1911], block_coordinates[1912], block_coordinates[1913], block_coordinates[1914], block_coordinates[1915], block_coordinates[1916], block_coordinates[1917], block_coordinates[1918], block_coordinates[1919], block_coordinates[1920], block_coordinates[1921], block_coordinates[1922], block_coordinates[1923], block_coordinates[1924], block_coordinates[1925], block_coordinates[1926], block_coordinates[1927], block_coordinates[1928], block_coordinates[1929], block_coordinates[1930], block_coordinates[1931], block_coordinates[1932], block_coordinates[1933], block_coordinates[1934], block_coordinates[1935], block_coordinates[1936], block_coordinates[1937], block_coordinates[1938], block_coordinates[1939], block_coordinates[1940], block_coordinates[1941], block_coordinates[1942], block_coordinates[1943], block_coordinates[1944], block_coordinates[1945], block_coordinates[1946], block_coordinates[1947], block_coordinates[1948], block_coordinates[1949], block_coordinates[1950], block_coordinates[1951], block_coordinates[1952], block_coordinates[1953], block_coordinates[1954], block_coordinates[1955], block_coordinates[1956], block_coordinates[1957], block_coordinates[1958], block_coordinates[1959], block_coordinates[1960], block_coordinates[1961], block_coordinates[1962], block_coordinates[1963], block_coordinates[1964], block_coordinates[1965], block_coordinates[1966], block_coordinates[1967], block_coordinates[1968], block_coordinates[1969], block_coordinates[1970], block_coordinates[1971], block_coordinates[1972], block_coordinates[1973], block_coordinates[1974], block_coordinates[1975], block_coordinates[1976], block_coordinates[1977], block_coordinates[1978], block_coordinates[1979], block_coordinates[1980], block_coordinates[1981], block_coordinates[1982], block_coordinates[1983], block_coordinates[1984], block_coordinates[1985], block_coordinates[1986], block_coordinates[1987], block_coordinates[1988], block_coordinates[1989], block_coordinates[1990], block_coordinates[1991], block_coordinates[1992], block_coordinates[1993], block_coordinates[1994], block_coordinates[1995], block_coordinates[1996], block_coordinates[1997], block_coordinates[1998], block_coordinates[1999], block_coordinates[2000], block_coordinates[2001], block_coordinates[2002], block_coordinates[2003], block_coordinates[2004], block_coordinates[2005], block_coordinates[2006], block_coordinates[2007], block_coordinates[2008], block_coordinates[2009], block_coordinates[2010], block_coordinates[2011], block_coordinates[2012], block_coordinates[2013], block_coordinates[2014], block_coordinates[2015], block_coordinates[2016], block_coordinates[2017], block_coordinates[2018], block_coordinates[2019], block_coordinates[2020], block_coordinates[2021], block_coordinates[2022], block_coordinates[2023], block_coordinates[2024], block_coordinates[2025], block_coordinates[2026], block_coordinates[2027], block_coordinates[2028], block_coordinates[2029], block_coordinates[2030], block_coordinates[2031], block_coordinates[2032], block_coordinates[2033], block_coordinates[2034], block_coordinates[2035], block_coordinates[2036], block_coordinates[2037], block_coordinates[2038], block_coordinates[2039], block_coordinates[2040], block_coordinates[2041], block_coordinates[2042], block_coordinates[2043], block_coordinates[2044], block_coordinates[2045], block_coordinates[2046], block_coordinates[2047], block_coordinates[2048], block_coordinates[2049], block_coordinates[2050], block_coordinates[2051], block_coordinates[2052], block_coordinates[2053], block_coordinates[2054], block_coordinates[2055], block_coordinates[2056], block_coordinates[2057], block_coordinates[2058], block_coordinates[2059], block_coordinates[2060], block_coordinates[2061], block_coordinates[2062], block_coordinates[2063], block_coordinates[2064], block_coordinates[2065], block_coordinates[2066], block_coordinates[2067], block_coordinates[2068], block_coordinates[2069], block_coordinates[2070], block_coordinates[2071], block_coordinates[2072], block_coordinates[2073], block_coordinates[2074], block_coordinates[2075], block_coordinates[2076], block_coordinates[2077], block_coordinates[2078], block_coordinates[2079], block_coordinates[2080], block_coordinates[2081], block_coordinates[2082], block_coordinates[2083], block_coordinates[2084], block_coordinates[2085], block_coordinates[2086], block_coordinates[2087], block_coordinates[2088], block_coordinates[2089], block_coordinates[2090], block_coordinates[2091], block_coordinates[2092], block_coordinates[2093], block_coordinates[2094], block_coordinates[2095], block_coordinates[2096], block_coordinates[2097], block_coordinates[2098], block_coordinates[2099], block_coordinates[2100], block_coordinates[2101], block_coordinates[2102], block_coordinates[2103], block_coordinates[2104], block_coordinates[2105], block_coordinates[2106], block_coordinates[2107], block_coordinates[2108], block_coordinates[2109], block_coordinates[2110], block_coordinates[2111], block_coordinates[2112], block_coordinates[2113], block_coordinates[2114], block_coordinates[2115], block_coordinates[2116], block_coordinates[2117], block_coordinates[2118], block_coordinates[2119], block_coordinates[2120], block_coordinates[2121], block_coordinates[2122], block_coordinates[2123], block_coordinates[2124], block_coordinates[2125], block_coordinates[2126], block_coordinates[2127], block_coordinates[2128], block_coordinates[2129], block_coordinates[2130], block_coordinates[2131], block_coordinates[2132], block_coordinates[2133], block_coordinates[2134], block_coordinates[2135], block_coordinates[2136], block_coordinates[2137], block_coordinates[2138], block_coordinates[2139], block_coordinates[2140], block_coordinates[2141], block_coordinates[2142], block_coordinates[2143], block_coordinates[2144], block_coordinates[2145], block_coordinates[2146], block_coordinates[2147], block_coordinates[2148], block_coordinates[2149], block_coordinates[2150], block_coordinates[2151], block_coordinates[2152], block_coordinates[2153], block_coordinates[2154], block_coordinates[2155], block_coordinates[2156], block_coordinates[2157], block_coordinates[2158], block_coordinates[2159], block_coordinates[2160], block_coordinates[2161], block_coordinates[2162], block_coordinates[2163], block_coordinates[2164], block_coordinates[2165], block_coordinates[2166], block_coordinates[2167], block_coordinates[2168], block_coordinates[2169], block_coordinates[2170], block_coordinates[2171], block_coordinates[2172], block_coordinates[2173], block_coordinates[2174], block_coordinates[2175], block_coordinates[2176], block_coordinates[2177], block_coordinates[2178], block_coordinates[2179], block_coordinates[2180], block_coordinates[2181], block_coordinates[2182], block_coordinates[2183], block_coordinates[2184], block_coordinates[2185], block_coordinates[2186]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t * block_current_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_current_streams[stream] = block_current[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
        }
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current_streams, kappa, block_output_streams);

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

extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex729_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex729_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex729_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex729_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex729_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[729],
        const double kappa,
        double *const SFEM_RESTRICT output[729]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<double, 729, 729, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<double>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[729],
        const float kappa,
        float *const SFEM_RESTRICT output[729]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<float, 729, 729, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<float>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex729_jacobian_action_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();

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

        for (int stream = 0; stream < 729; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t * block_direction_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_direction_streams[stream] = block_direction[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
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
        const scalar_t *const block_adjugate[9] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_direction_streams, kappa, block_output_streams);

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

extern "C" int laplace_proteus_hex729_jacobian_action_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex729_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_jacobian_action_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex729_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();

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

        for (int stream = 0; stream < 729; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191], block_coordinates[192], block_coordinates[193], block_coordinates[194], block_coordinates[195], block_coordinates[196], block_coordinates[197], block_coordinates[198], block_coordinates[199], block_coordinates[200], block_coordinates[201], block_coordinates[202], block_coordinates[203], block_coordinates[204], block_coordinates[205], block_coordinates[206], block_coordinates[207], block_coordinates[208], block_coordinates[209], block_coordinates[210], block_coordinates[211], block_coordinates[212], block_coordinates[213], block_coordinates[214], block_coordinates[215], block_coordinates[216], block_coordinates[217], block_coordinates[218], block_coordinates[219], block_coordinates[220], block_coordinates[221], block_coordinates[222], block_coordinates[223], block_coordinates[224], block_coordinates[225], block_coordinates[226], block_coordinates[227], block_coordinates[228], block_coordinates[229], block_coordinates[230], block_coordinates[231], block_coordinates[232], block_coordinates[233], block_coordinates[234], block_coordinates[235], block_coordinates[236], block_coordinates[237], block_coordinates[238], block_coordinates[239], block_coordinates[240], block_coordinates[241], block_coordinates[242], block_coordinates[243], block_coordinates[244], block_coordinates[245], block_coordinates[246], block_coordinates[247], block_coordinates[248], block_coordinates[249], block_coordinates[250], block_coordinates[251], block_coordinates[252], block_coordinates[253], block_coordinates[254], block_coordinates[255], block_coordinates[256], block_coordinates[257], block_coordinates[258], block_coordinates[259], block_coordinates[260], block_coordinates[261], block_coordinates[262], block_coordinates[263], block_coordinates[264], block_coordinates[265], block_coordinates[266], block_coordinates[267], block_coordinates[268], block_coordinates[269], block_coordinates[270], block_coordinates[271], block_coordinates[272], block_coordinates[273], block_coordinates[274], block_coordinates[275], block_coordinates[276], block_coordinates[277], block_coordinates[278], block_coordinates[279], block_coordinates[280], block_coordinates[281], block_coordinates[282], block_coordinates[283], block_coordinates[284], block_coordinates[285], block_coordinates[286], block_coordinates[287], block_coordinates[288], block_coordinates[289], block_coordinates[290], block_coordinates[291], block_coordinates[292], block_coordinates[293], block_coordinates[294], block_coordinates[295], block_coordinates[296], block_coordinates[297], block_coordinates[298], block_coordinates[299], block_coordinates[300], block_coordinates[301], block_coordinates[302], block_coordinates[303], block_coordinates[304], block_coordinates[305], block_coordinates[306], block_coordinates[307], block_coordinates[308], block_coordinates[309], block_coordinates[310], block_coordinates[311], block_coordinates[312], block_coordinates[313], block_coordinates[314], block_coordinates[315], block_coordinates[316], block_coordinates[317], block_coordinates[318], block_coordinates[319], block_coordinates[320], block_coordinates[321], block_coordinates[322], block_coordinates[323], block_coordinates[324], block_coordinates[325], block_coordinates[326], block_coordinates[327], block_coordinates[328], block_coordinates[329], block_coordinates[330], block_coordinates[331], block_coordinates[332], block_coordinates[333], block_coordinates[334], block_coordinates[335], block_coordinates[336], block_coordinates[337], block_coordinates[338], block_coordinates[339], block_coordinates[340], block_coordinates[341], block_coordinates[342], block_coordinates[343], block_coordinates[344], block_coordinates[345], block_coordinates[346], block_coordinates[347], block_coordinates[348], block_coordinates[349], block_coordinates[350], block_coordinates[351], block_coordinates[352], block_coordinates[353], block_coordinates[354], block_coordinates[355], block_coordinates[356], block_coordinates[357], block_coordinates[358], block_coordinates[359], block_coordinates[360], block_coordinates[361], block_coordinates[362], block_coordinates[363], block_coordinates[364], block_coordinates[365], block_coordinates[366], block_coordinates[367], block_coordinates[368], block_coordinates[369], block_coordinates[370], block_coordinates[371], block_coordinates[372], block_coordinates[373], block_coordinates[374], block_coordinates[375], block_coordinates[376], block_coordinates[377], block_coordinates[378], block_coordinates[379], block_coordinates[380], block_coordinates[381], block_coordinates[382], block_coordinates[383], block_coordinates[384], block_coordinates[385], block_coordinates[386], block_coordinates[387], block_coordinates[388], block_coordinates[389], block_coordinates[390], block_coordinates[391], block_coordinates[392], block_coordinates[393], block_coordinates[394], block_coordinates[395], block_coordinates[396], block_coordinates[397], block_coordinates[398], block_coordinates[399], block_coordinates[400], block_coordinates[401], block_coordinates[402], block_coordinates[403], block_coordinates[404], block_coordinates[405], block_coordinates[406], block_coordinates[407], block_coordinates[408], block_coordinates[409], block_coordinates[410], block_coordinates[411], block_coordinates[412], block_coordinates[413], block_coordinates[414], block_coordinates[415], block_coordinates[416], block_coordinates[417], block_coordinates[418], block_coordinates[419], block_coordinates[420], block_coordinates[421], block_coordinates[422], block_coordinates[423], block_coordinates[424], block_coordinates[425], block_coordinates[426], block_coordinates[427], block_coordinates[428], block_coordinates[429], block_coordinates[430], block_coordinates[431], block_coordinates[432], block_coordinates[433], block_coordinates[434], block_coordinates[435], block_coordinates[436], block_coordinates[437], block_coordinates[438], block_coordinates[439], block_coordinates[440], block_coordinates[441], block_coordinates[442], block_coordinates[443], block_coordinates[444], block_coordinates[445], block_coordinates[446], block_coordinates[447], block_coordinates[448], block_coordinates[449], block_coordinates[450], block_coordinates[451], block_coordinates[452], block_coordinates[453], block_coordinates[454], block_coordinates[455], block_coordinates[456], block_coordinates[457], block_coordinates[458], block_coordinates[459], block_coordinates[460], block_coordinates[461], block_coordinates[462], block_coordinates[463], block_coordinates[464], block_coordinates[465], block_coordinates[466], block_coordinates[467], block_coordinates[468], block_coordinates[469], block_coordinates[470], block_coordinates[471], block_coordinates[472], block_coordinates[473], block_coordinates[474], block_coordinates[475], block_coordinates[476], block_coordinates[477], block_coordinates[478], block_coordinates[479], block_coordinates[480], block_coordinates[481], block_coordinates[482], block_coordinates[483], block_coordinates[484], block_coordinates[485], block_coordinates[486], block_coordinates[487], block_coordinates[488], block_coordinates[489], block_coordinates[490], block_coordinates[491], block_coordinates[492], block_coordinates[493], block_coordinates[494], block_coordinates[495], block_coordinates[496], block_coordinates[497], block_coordinates[498], block_coordinates[499], block_coordinates[500], block_coordinates[501], block_coordinates[502], block_coordinates[503], block_coordinates[504], block_coordinates[505], block_coordinates[506], block_coordinates[507], block_coordinates[508], block_coordinates[509], block_coordinates[510], block_coordinates[511], block_coordinates[512], block_coordinates[513], block_coordinates[514], block_coordinates[515], block_coordinates[516], block_coordinates[517], block_coordinates[518], block_coordinates[519], block_coordinates[520], block_coordinates[521], block_coordinates[522], block_coordinates[523], block_coordinates[524], block_coordinates[525], block_coordinates[526], block_coordinates[527], block_coordinates[528], block_coordinates[529], block_coordinates[530], block_coordinates[531], block_coordinates[532], block_coordinates[533], block_coordinates[534], block_coordinates[535], block_coordinates[536], block_coordinates[537], block_coordinates[538], block_coordinates[539], block_coordinates[540], block_coordinates[541], block_coordinates[542], block_coordinates[543], block_coordinates[544], block_coordinates[545], block_coordinates[546], block_coordinates[547], block_coordinates[548], block_coordinates[549], block_coordinates[550], block_coordinates[551], block_coordinates[552], block_coordinates[553], block_coordinates[554], block_coordinates[555], block_coordinates[556], block_coordinates[557], block_coordinates[558], block_coordinates[559], block_coordinates[560], block_coordinates[561], block_coordinates[562], block_coordinates[563], block_coordinates[564], block_coordinates[565], block_coordinates[566], block_coordinates[567], block_coordinates[568], block_coordinates[569], block_coordinates[570], block_coordinates[571], block_coordinates[572], block_coordinates[573], block_coordinates[574], block_coordinates[575], block_coordinates[576], block_coordinates[577], block_coordinates[578], block_coordinates[579], block_coordinates[580], block_coordinates[581], block_coordinates[582], block_coordinates[583], block_coordinates[584], block_coordinates[585], block_coordinates[586], block_coordinates[587], block_coordinates[588], block_coordinates[589], block_coordinates[590], block_coordinates[591], block_coordinates[592], block_coordinates[593], block_coordinates[594], block_coordinates[595], block_coordinates[596], block_coordinates[597], block_coordinates[598], block_coordinates[599], block_coordinates[600], block_coordinates[601], block_coordinates[602], block_coordinates[603], block_coordinates[604], block_coordinates[605], block_coordinates[606], block_coordinates[607], block_coordinates[608], block_coordinates[609], block_coordinates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613], block_coordinates[614], block_coordinates[615], block_coordinates[616], block_coordinates[617], block_coordinates[618], block_coordinates[619], block_coordinates[620], block_coordinates[621], block_coordinates[622], block_coordinates[623], block_coordinates[624], block_coordinates[625], block_coordinates[626], block_coordinates[627], block_coordinates[628], block_coordinates[629], block_coordinates[630], block_coordinates[631], block_coordinates[632], block_coordinates[633], block_coordinates[634], block_coordinates[635], block_coordinates[636], block_coordinates[637], block_coordinates[638], block_coordinates[639], block_coordinates[640], block_coordinates[641], block_coordinates[642], block_coordinates[643], block_coordinates[644], block_coordinates[645], block_coordinates[646], block_coordinates[647], block_coordinates[648], block_coordinates[649], block_coordinates[650], block_coordinates[651], block_coordinates[652], block_coordinates[653], block_coordinates[654], block_coordinates[655], block_coordinates[656], block_coordinates[657], block_coordinates[658], block_coordinates[659], block_coordinates[660], block_coordinates[661], block_coordinates[662], block_coordinates[663], block_coordinates[664], block_coordinates[665], block_coordinates[666], block_coordinates[667], block_coordinates[668], block_coordinates[669], block_coordinates[670], block_coordinates[671], block_coordinates[672], block_coordinates[673], block_coordinates[674], block_coordinates[675], block_coordinates[676], block_coordinates[677], block_coordinates[678], block_coordinates[679], block_coordinates[680], block_coordinates[681], block_coordinates[682], block_coordinates[683], block_coordinates[684], block_coordinates[685], block_coordinates[686], block_coordinates[687], block_coordinates[688], block_coordinates[689], block_coordinates[690], block_coordinates[691], block_coordinates[692], block_coordinates[693], block_coordinates[694], block_coordinates[695], block_coordinates[696], block_coordinates[697], block_coordinates[698], block_coordinates[699], block_coordinates[700], block_coordinates[701], block_coordinates[702], block_coordinates[703], block_coordinates[704], block_coordinates[705], block_coordinates[706], block_coordinates[707], block_coordinates[708], block_coordinates[709], block_coordinates[710], block_coordinates[711], block_coordinates[712], block_coordinates[713], block_coordinates[714], block_coordinates[715], block_coordinates[716], block_coordinates[717], block_coordinates[718], block_coordinates[719], block_coordinates[720], block_coordinates[721], block_coordinates[722], block_coordinates[723], block_coordinates[724], block_coordinates[725], block_coordinates[726], block_coordinates[727], block_coordinates[728], block_coordinates[729], block_coordinates[730], block_coordinates[731], block_coordinates[732], block_coordinates[733], block_coordinates[734], block_coordinates[735], block_coordinates[736], block_coordinates[737], block_coordinates[738], block_coordinates[739], block_coordinates[740], block_coordinates[741], block_coordinates[742], block_coordinates[743], block_coordinates[744], block_coordinates[745], block_coordinates[746], block_coordinates[747], block_coordinates[748], block_coordinates[749], block_coordinates[750], block_coordinates[751], block_coordinates[752], block_coordinates[753], block_coordinates[754], block_coordinates[755], block_coordinates[756], block_coordinates[757], block_coordinates[758], block_coordinates[759], block_coordinates[760], block_coordinates[761], block_coordinates[762], block_coordinates[763], block_coordinates[764], block_coordinates[765], block_coordinates[766], block_coordinates[767], block_coordinates[768], block_coordinates[769], block_coordinates[770], block_coordinates[771], block_coordinates[772], block_coordinates[773], block_coordinates[774], block_coordinates[775], block_coordinates[776], block_coordinates[777], block_coordinates[778], block_coordinates[779], block_coordinates[780], block_coordinates[781], block_coordinates[782], block_coordinates[783], block_coordinates[784], block_coordinates[785], block_coordinates[786], block_coordinates[787], block_coordinates[788], block_coordinates[789], block_coordinates[790], block_coordinates[791], block_coordinates[792], block_coordinates[793], block_coordinates[794], block_coordinates[795], block_coordinates[796], block_coordinates[797], block_coordinates[798], block_coordinates[799], block_coordinates[800], block_coordinates[801], block_coordinates[802], block_coordinates[803], block_coordinates[804], block_coordinates[805], block_coordinates[806], block_coordinates[807], block_coordinates[808], block_coordinates[809], block_coordinates[810], block_coordinates[811], block_coordinates[812], block_coordinates[813], block_coordinates[814], block_coordinates[815], block_coordinates[816], block_coordinates[817], block_coordinates[818], block_coordinates[819], block_coordinates[820], block_coordinates[821], block_coordinates[822], block_coordinates[823], block_coordinates[824], block_coordinates[825], block_coordinates[826], block_coordinates[827], block_coordinates[828], block_coordinates[829], block_coordinates[830], block_coordinates[831], block_coordinates[832], block_coordinates[833], block_coordinates[834], block_coordinates[835], block_coordinates[836], block_coordinates[837], block_coordinates[838], block_coordinates[839], block_coordinates[840], block_coordinates[841], block_coordinates[842], block_coordinates[843], block_coordinates[844], block_coordinates[845], block_coordinates[846], block_coordinates[847], block_coordinates[848], block_coordinates[849], block_coordinates[850], block_coordinates[851], block_coordinates[852], block_coordinates[853], block_coordinates[854], block_coordinates[855], block_coordinates[856], block_coordinates[857], block_coordinates[858], block_coordinates[859], block_coordinates[860], block_coordinates[861], block_coordinates[862], block_coordinates[863], block_coordinates[864], block_coordinates[865], block_coordinates[866], block_coordinates[867], block_coordinates[868], block_coordinates[869], block_coordinates[870], block_coordinates[871], block_coordinates[872], block_coordinates[873], block_coordinates[874], block_coordinates[875], block_coordinates[876], block_coordinates[877], block_coordinates[878], block_coordinates[879], block_coordinates[880], block_coordinates[881], block_coordinates[882], block_coordinates[883], block_coordinates[884], block_coordinates[885], block_coordinates[886], block_coordinates[887], block_coordinates[888], block_coordinates[889], block_coordinates[890], block_coordinates[891], block_coordinates[892], block_coordinates[893], block_coordinates[894], block_coordinates[895], block_coordinates[896], block_coordinates[897], block_coordinates[898], block_coordinates[899], block_coordinates[900], block_coordinates[901], block_coordinates[902], block_coordinates[903], block_coordinates[904], block_coordinates[905], block_coordinates[906], block_coordinates[907], block_coordinates[908], block_coordinates[909], block_coordinates[910], block_coordinates[911], block_coordinates[912], block_coordinates[913], block_coordinates[914], block_coordinates[915], block_coordinates[916], block_coordinates[917], block_coordinates[918], block_coordinates[919], block_coordinates[920], block_coordinates[921], block_coordinates[922], block_coordinates[923], block_coordinates[924], block_coordinates[925], block_coordinates[926], block_coordinates[927], block_coordinates[928], block_coordinates[929], block_coordinates[930], block_coordinates[931], block_coordinates[932], block_coordinates[933], block_coordinates[934], block_coordinates[935], block_coordinates[936], block_coordinates[937], block_coordinates[938], block_coordinates[939], block_coordinates[940], block_coordinates[941], block_coordinates[942], block_coordinates[943], block_coordinates[944], block_coordinates[945], block_coordinates[946], block_coordinates[947], block_coordinates[948], block_coordinates[949], block_coordinates[950], block_coordinates[951], block_coordinates[952], block_coordinates[953], block_coordinates[954], block_coordinates[955], block_coordinates[956], block_coordinates[957], block_coordinates[958], block_coordinates[959], block_coordinates[960], block_coordinates[961], block_coordinates[962], block_coordinates[963], block_coordinates[964], block_coordinates[965], block_coordinates[966], block_coordinates[967], block_coordinates[968], block_coordinates[969], block_coordinates[970], block_coordinates[971], block_coordinates[972], block_coordinates[973], block_coordinates[974], block_coordinates[975], block_coordinates[976], block_coordinates[977], block_coordinates[978], block_coordinates[979], block_coordinates[980], block_coordinates[981], block_coordinates[982], block_coordinates[983], block_coordinates[984], block_coordinates[985], block_coordinates[986], block_coordinates[987], block_coordinates[988], block_coordinates[989], block_coordinates[990], block_coordinates[991], block_coordinates[992], block_coordinates[993], block_coordinates[994], block_coordinates[995], block_coordinates[996], block_coordinates[997], block_coordinates[998], block_coordinates[999], block_coordinates[1000], block_coordinates[1001], block_coordinates[1002], block_coordinates[1003], block_coordinates[1004], block_coordinates[1005], block_coordinates[1006], block_coordinates[1007], block_coordinates[1008], block_coordinates[1009], block_coordinates[1010], block_coordinates[1011], block_coordinates[1012], block_coordinates[1013], block_coordinates[1014], block_coordinates[1015], block_coordinates[1016], block_coordinates[1017], block_coordinates[1018], block_coordinates[1019], block_coordinates[1020], block_coordinates[1021], block_coordinates[1022], block_coordinates[1023], block_coordinates[1024], block_coordinates[1025], block_coordinates[1026], block_coordinates[1027], block_coordinates[1028], block_coordinates[1029], block_coordinates[1030], block_coordinates[1031], block_coordinates[1032], block_coordinates[1033], block_coordinates[1034], block_coordinates[1035], block_coordinates[1036], block_coordinates[1037], block_coordinates[1038], block_coordinates[1039], block_coordinates[1040], block_coordinates[1041], block_coordinates[1042], block_coordinates[1043], block_coordinates[1044], block_coordinates[1045], block_coordinates[1046], block_coordinates[1047], block_coordinates[1048], block_coordinates[1049], block_coordinates[1050], block_coordinates[1051], block_coordinates[1052], block_coordinates[1053], block_coordinates[1054], block_coordinates[1055], block_coordinates[1056], block_coordinates[1057], block_coordinates[1058], block_coordinates[1059], block_coordinates[1060], block_coordinates[1061], block_coordinates[1062], block_coordinates[1063], block_coordinates[1064], block_coordinates[1065], block_coordinates[1066], block_coordinates[1067], block_coordinates[1068], block_coordinates[1069], block_coordinates[1070], block_coordinates[1071], block_coordinates[1072], block_coordinates[1073], block_coordinates[1074], block_coordinates[1075], block_coordinates[1076], block_coordinates[1077], block_coordinates[1078], block_coordinates[1079], block_coordinates[1080], block_coordinates[1081], block_coordinates[1082], block_coordinates[1083], block_coordinates[1084], block_coordinates[1085], block_coordinates[1086], block_coordinates[1087], block_coordinates[1088], block_coordinates[1089], block_coordinates[1090], block_coordinates[1091], block_coordinates[1092], block_coordinates[1093], block_coordinates[1094], block_coordinates[1095], block_coordinates[1096], block_coordinates[1097], block_coordinates[1098], block_coordinates[1099], block_coordinates[1100], block_coordinates[1101], block_coordinates[1102], block_coordinates[1103], block_coordinates[1104], block_coordinates[1105], block_coordinates[1106], block_coordinates[1107], block_coordinates[1108], block_coordinates[1109], block_coordinates[1110], block_coordinates[1111], block_coordinates[1112], block_coordinates[1113], block_coordinates[1114], block_coordinates[1115], block_coordinates[1116], block_coordinates[1117], block_coordinates[1118], block_coordinates[1119], block_coordinates[1120], block_coordinates[1121], block_coordinates[1122], block_coordinates[1123], block_coordinates[1124], block_coordinates[1125], block_coordinates[1126], block_coordinates[1127], block_coordinates[1128], block_coordinates[1129], block_coordinates[1130], block_coordinates[1131], block_coordinates[1132], block_coordinates[1133], block_coordinates[1134], block_coordinates[1135], block_coordinates[1136], block_coordinates[1137], block_coordinates[1138], block_coordinates[1139], block_coordinates[1140], block_coordinates[1141], block_coordinates[1142], block_coordinates[1143], block_coordinates[1144], block_coordinates[1145], block_coordinates[1146], block_coordinates[1147], block_coordinates[1148], block_coordinates[1149], block_coordinates[1150], block_coordinates[1151], block_coordinates[1152], block_coordinates[1153], block_coordinates[1154], block_coordinates[1155], block_coordinates[1156], block_coordinates[1157], block_coordinates[1158], block_coordinates[1159], block_coordinates[1160], block_coordinates[1161], block_coordinates[1162], block_coordinates[1163], block_coordinates[1164], block_coordinates[1165], block_coordinates[1166], block_coordinates[1167], block_coordinates[1168], block_coordinates[1169], block_coordinates[1170], block_coordinates[1171], block_coordinates[1172], block_coordinates[1173], block_coordinates[1174], block_coordinates[1175], block_coordinates[1176], block_coordinates[1177], block_coordinates[1178], block_coordinates[1179], block_coordinates[1180], block_coordinates[1181], block_coordinates[1182], block_coordinates[1183], block_coordinates[1184], block_coordinates[1185], block_coordinates[1186], block_coordinates[1187], block_coordinates[1188], block_coordinates[1189], block_coordinates[1190], block_coordinates[1191], block_coordinates[1192], block_coordinates[1193], block_coordinates[1194], block_coordinates[1195], block_coordinates[1196], block_coordinates[1197], block_coordinates[1198], block_coordinates[1199], block_coordinates[1200], block_coordinates[1201], block_coordinates[1202], block_coordinates[1203], block_coordinates[1204], block_coordinates[1205], block_coordinates[1206], block_coordinates[1207], block_coordinates[1208], block_coordinates[1209], block_coordinates[1210], block_coordinates[1211], block_coordinates[1212], block_coordinates[1213], block_coordinates[1214], block_coordinates[1215], block_coordinates[1216], block_coordinates[1217], block_coordinates[1218], block_coordinates[1219], block_coordinates[1220], block_coordinates[1221], block_coordinates[1222], block_coordinates[1223], block_coordinates[1224], block_coordinates[1225], block_coordinates[1226], block_coordinates[1227], block_coordinates[1228], block_coordinates[1229], block_coordinates[1230], block_coordinates[1231], block_coordinates[1232], block_coordinates[1233], block_coordinates[1234], block_coordinates[1235], block_coordinates[1236], block_coordinates[1237], block_coordinates[1238], block_coordinates[1239], block_coordinates[1240], block_coordinates[1241], block_coordinates[1242], block_coordinates[1243], block_coordinates[1244], block_coordinates[1245], block_coordinates[1246], block_coordinates[1247], block_coordinates[1248], block_coordinates[1249], block_coordinates[1250], block_coordinates[1251], block_coordinates[1252], block_coordinates[1253], block_coordinates[1254], block_coordinates[1255], block_coordinates[1256], block_coordinates[1257], block_coordinates[1258], block_coordinates[1259], block_coordinates[1260], block_coordinates[1261], block_coordinates[1262], block_coordinates[1263], block_coordinates[1264], block_coordinates[1265], block_coordinates[1266], block_coordinates[1267], block_coordinates[1268], block_coordinates[1269], block_coordinates[1270], block_coordinates[1271], block_coordinates[1272], block_coordinates[1273], block_coordinates[1274], block_coordinates[1275], block_coordinates[1276], block_coordinates[1277], block_coordinates[1278], block_coordinates[1279], block_coordinates[1280], block_coordinates[1281], block_coordinates[1282], block_coordinates[1283], block_coordinates[1284], block_coordinates[1285], block_coordinates[1286], block_coordinates[1287], block_coordinates[1288], block_coordinates[1289], block_coordinates[1290], block_coordinates[1291], block_coordinates[1292], block_coordinates[1293], block_coordinates[1294], block_coordinates[1295], block_coordinates[1296], block_coordinates[1297], block_coordinates[1298], block_coordinates[1299], block_coordinates[1300], block_coordinates[1301], block_coordinates[1302], block_coordinates[1303], block_coordinates[1304], block_coordinates[1305], block_coordinates[1306], block_coordinates[1307], block_coordinates[1308], block_coordinates[1309], block_coordinates[1310], block_coordinates[1311], block_coordinates[1312], block_coordinates[1313], block_coordinates[1314], block_coordinates[1315], block_coordinates[1316], block_coordinates[1317], block_coordinates[1318], block_coordinates[1319], block_coordinates[1320], block_coordinates[1321], block_coordinates[1322], block_coordinates[1323], block_coordinates[1324], block_coordinates[1325], block_coordinates[1326], block_coordinates[1327], block_coordinates[1328], block_coordinates[1329], block_coordinates[1330], block_coordinates[1331], block_coordinates[1332], block_coordinates[1333], block_coordinates[1334], block_coordinates[1335], block_coordinates[1336], block_coordinates[1337], block_coordinates[1338], block_coordinates[1339], block_coordinates[1340], block_coordinates[1341], block_coordinates[1342], block_coordinates[1343], block_coordinates[1344], block_coordinates[1345], block_coordinates[1346], block_coordinates[1347], block_coordinates[1348], block_coordinates[1349], block_coordinates[1350], block_coordinates[1351], block_coordinates[1352], block_coordinates[1353], block_coordinates[1354], block_coordinates[1355], block_coordinates[1356], block_coordinates[1357], block_coordinates[1358], block_coordinates[1359], block_coordinates[1360], block_coordinates[1361], block_coordinates[1362], block_coordinates[1363], block_coordinates[1364], block_coordinates[1365], block_coordinates[1366], block_coordinates[1367], block_coordinates[1368], block_coordinates[1369], block_coordinates[1370], block_coordinates[1371], block_coordinates[1372], block_coordinates[1373], block_coordinates[1374], block_coordinates[1375], block_coordinates[1376], block_coordinates[1377], block_coordinates[1378], block_coordinates[1379], block_coordinates[1380], block_coordinates[1381], block_coordinates[1382], block_coordinates[1383], block_coordinates[1384], block_coordinates[1385], block_coordinates[1386], block_coordinates[1387], block_coordinates[1388], block_coordinates[1389], block_coordinates[1390], block_coordinates[1391], block_coordinates[1392], block_coordinates[1393], block_coordinates[1394], block_coordinates[1395], block_coordinates[1396], block_coordinates[1397], block_coordinates[1398], block_coordinates[1399], block_coordinates[1400], block_coordinates[1401], block_coordinates[1402], block_coordinates[1403], block_coordinates[1404], block_coordinates[1405], block_coordinates[1406], block_coordinates[1407], block_coordinates[1408], block_coordinates[1409], block_coordinates[1410], block_coordinates[1411], block_coordinates[1412], block_coordinates[1413], block_coordinates[1414], block_coordinates[1415], block_coordinates[1416], block_coordinates[1417], block_coordinates[1418], block_coordinates[1419], block_coordinates[1420], block_coordinates[1421], block_coordinates[1422], block_coordinates[1423], block_coordinates[1424], block_coordinates[1425], block_coordinates[1426], block_coordinates[1427], block_coordinates[1428], block_coordinates[1429], block_coordinates[1430], block_coordinates[1431], block_coordinates[1432], block_coordinates[1433], block_coordinates[1434], block_coordinates[1435], block_coordinates[1436], block_coordinates[1437], block_coordinates[1438], block_coordinates[1439], block_coordinates[1440], block_coordinates[1441], block_coordinates[1442], block_coordinates[1443], block_coordinates[1444], block_coordinates[1445], block_coordinates[1446], block_coordinates[1447], block_coordinates[1448], block_coordinates[1449], block_coordinates[1450], block_coordinates[1451], block_coordinates[1452], block_coordinates[1453], block_coordinates[1454], block_coordinates[1455], block_coordinates[1456], block_coordinates[1457], block_coordinates[1458], block_coordinates[1459], block_coordinates[1460], block_coordinates[1461], block_coordinates[1462], block_coordinates[1463], block_coordinates[1464], block_coordinates[1465], block_coordinates[1466], block_coordinates[1467], block_coordinates[1468], block_coordinates[1469], block_coordinates[1470], block_coordinates[1471], block_coordinates[1472], block_coordinates[1473], block_coordinates[1474], block_coordinates[1475], block_coordinates[1476], block_coordinates[1477], block_coordinates[1478], block_coordinates[1479], block_coordinates[1480], block_coordinates[1481], block_coordinates[1482], block_coordinates[1483], block_coordinates[1484], block_coordinates[1485], block_coordinates[1486], block_coordinates[1487], block_coordinates[1488], block_coordinates[1489], block_coordinates[1490], block_coordinates[1491], block_coordinates[1492], block_coordinates[1493], block_coordinates[1494], block_coordinates[1495], block_coordinates[1496], block_coordinates[1497], block_coordinates[1498], block_coordinates[1499], block_coordinates[1500], block_coordinates[1501], block_coordinates[1502], block_coordinates[1503], block_coordinates[1504], block_coordinates[1505], block_coordinates[1506], block_coordinates[1507], block_coordinates[1508], block_coordinates[1509], block_coordinates[1510], block_coordinates[1511], block_coordinates[1512], block_coordinates[1513], block_coordinates[1514], block_coordinates[1515], block_coordinates[1516], block_coordinates[1517], block_coordinates[1518], block_coordinates[1519], block_coordinates[1520], block_coordinates[1521], block_coordinates[1522], block_coordinates[1523], block_coordinates[1524], block_coordinates[1525], block_coordinates[1526], block_coordinates[1527], block_coordinates[1528], block_coordinates[1529], block_coordinates[1530], block_coordinates[1531], block_coordinates[1532], block_coordinates[1533], block_coordinates[1534], block_coordinates[1535], block_coordinates[1536], block_coordinates[1537], block_coordinates[1538], block_coordinates[1539], block_coordinates[1540], block_coordinates[1541], block_coordinates[1542], block_coordinates[1543], block_coordinates[1544], block_coordinates[1545], block_coordinates[1546], block_coordinates[1547], block_coordinates[1548], block_coordinates[1549], block_coordinates[1550], block_coordinates[1551], block_coordinates[1552], block_coordinates[1553], block_coordinates[1554], block_coordinates[1555], block_coordinates[1556], block_coordinates[1557], block_coordinates[1558], block_coordinates[1559], block_coordinates[1560], block_coordinates[1561], block_coordinates[1562], block_coordinates[1563], block_coordinates[1564], block_coordinates[1565], block_coordinates[1566], block_coordinates[1567], block_coordinates[1568], block_coordinates[1569], block_coordinates[1570], block_coordinates[1571], block_coordinates[1572], block_coordinates[1573], block_coordinates[1574], block_coordinates[1575], block_coordinates[1576], block_coordinates[1577], block_coordinates[1578], block_coordinates[1579], block_coordinates[1580], block_coordinates[1581], block_coordinates[1582], block_coordinates[1583], block_coordinates[1584], block_coordinates[1585], block_coordinates[1586], block_coordinates[1587], block_coordinates[1588], block_coordinates[1589], block_coordinates[1590], block_coordinates[1591], block_coordinates[1592], block_coordinates[1593], block_coordinates[1594], block_coordinates[1595], block_coordinates[1596], block_coordinates[1597], block_coordinates[1598], block_coordinates[1599], block_coordinates[1600], block_coordinates[1601], block_coordinates[1602], block_coordinates[1603], block_coordinates[1604], block_coordinates[1605], block_coordinates[1606], block_coordinates[1607], block_coordinates[1608], block_coordinates[1609], block_coordinates[1610], block_coordinates[1611], block_coordinates[1612], block_coordinates[1613], block_coordinates[1614], block_coordinates[1615], block_coordinates[1616], block_coordinates[1617], block_coordinates[1618], block_coordinates[1619], block_coordinates[1620], block_coordinates[1621], block_coordinates[1622], block_coordinates[1623], block_coordinates[1624], block_coordinates[1625], block_coordinates[1626], block_coordinates[1627], block_coordinates[1628], block_coordinates[1629], block_coordinates[1630], block_coordinates[1631], block_coordinates[1632], block_coordinates[1633], block_coordinates[1634], block_coordinates[1635], block_coordinates[1636], block_coordinates[1637], block_coordinates[1638], block_coordinates[1639], block_coordinates[1640], block_coordinates[1641], block_coordinates[1642], block_coordinates[1643], block_coordinates[1644], block_coordinates[1645], block_coordinates[1646], block_coordinates[1647], block_coordinates[1648], block_coordinates[1649], block_coordinates[1650], block_coordinates[1651], block_coordinates[1652], block_coordinates[1653], block_coordinates[1654], block_coordinates[1655], block_coordinates[1656], block_coordinates[1657], block_coordinates[1658], block_coordinates[1659], block_coordinates[1660], block_coordinates[1661], block_coordinates[1662], block_coordinates[1663], block_coordinates[1664], block_coordinates[1665], block_coordinates[1666], block_coordinates[1667], block_coordinates[1668], block_coordinates[1669], block_coordinates[1670], block_coordinates[1671], block_coordinates[1672], block_coordinates[1673], block_coordinates[1674], block_coordinates[1675], block_coordinates[1676], block_coordinates[1677], block_coordinates[1678], block_coordinates[1679], block_coordinates[1680], block_coordinates[1681], block_coordinates[1682], block_coordinates[1683], block_coordinates[1684], block_coordinates[1685], block_coordinates[1686], block_coordinates[1687], block_coordinates[1688], block_coordinates[1689], block_coordinates[1690], block_coordinates[1691], block_coordinates[1692], block_coordinates[1693], block_coordinates[1694], block_coordinates[1695], block_coordinates[1696], block_coordinates[1697], block_coordinates[1698], block_coordinates[1699], block_coordinates[1700], block_coordinates[1701], block_coordinates[1702], block_coordinates[1703], block_coordinates[1704], block_coordinates[1705], block_coordinates[1706], block_coordinates[1707], block_coordinates[1708], block_coordinates[1709], block_coordinates[1710], block_coordinates[1711], block_coordinates[1712], block_coordinates[1713], block_coordinates[1714], block_coordinates[1715], block_coordinates[1716], block_coordinates[1717], block_coordinates[1718], block_coordinates[1719], block_coordinates[1720], block_coordinates[1721], block_coordinates[1722], block_coordinates[1723], block_coordinates[1724], block_coordinates[1725], block_coordinates[1726], block_coordinates[1727], block_coordinates[1728], block_coordinates[1729], block_coordinates[1730], block_coordinates[1731], block_coordinates[1732], block_coordinates[1733], block_coordinates[1734], block_coordinates[1735], block_coordinates[1736], block_coordinates[1737], block_coordinates[1738], block_coordinates[1739], block_coordinates[1740], block_coordinates[1741], block_coordinates[1742], block_coordinates[1743], block_coordinates[1744], block_coordinates[1745], block_coordinates[1746], block_coordinates[1747], block_coordinates[1748], block_coordinates[1749], block_coordinates[1750], block_coordinates[1751], block_coordinates[1752], block_coordinates[1753], block_coordinates[1754], block_coordinates[1755], block_coordinates[1756], block_coordinates[1757], block_coordinates[1758], block_coordinates[1759], block_coordinates[1760], block_coordinates[1761], block_coordinates[1762], block_coordinates[1763], block_coordinates[1764], block_coordinates[1765], block_coordinates[1766], block_coordinates[1767], block_coordinates[1768], block_coordinates[1769], block_coordinates[1770], block_coordinates[1771], block_coordinates[1772], block_coordinates[1773], block_coordinates[1774], block_coordinates[1775], block_coordinates[1776], block_coordinates[1777], block_coordinates[1778], block_coordinates[1779], block_coordinates[1780], block_coordinates[1781], block_coordinates[1782], block_coordinates[1783], block_coordinates[1784], block_coordinates[1785], block_coordinates[1786], block_coordinates[1787], block_coordinates[1788], block_coordinates[1789], block_coordinates[1790], block_coordinates[1791], block_coordinates[1792], block_coordinates[1793], block_coordinates[1794], block_coordinates[1795], block_coordinates[1796], block_coordinates[1797], block_coordinates[1798], block_coordinates[1799], block_coordinates[1800], block_coordinates[1801], block_coordinates[1802], block_coordinates[1803], block_coordinates[1804], block_coordinates[1805], block_coordinates[1806], block_coordinates[1807], block_coordinates[1808], block_coordinates[1809], block_coordinates[1810], block_coordinates[1811], block_coordinates[1812], block_coordinates[1813], block_coordinates[1814], block_coordinates[1815], block_coordinates[1816], block_coordinates[1817], block_coordinates[1818], block_coordinates[1819], block_coordinates[1820], block_coordinates[1821], block_coordinates[1822], block_coordinates[1823], block_coordinates[1824], block_coordinates[1825], block_coordinates[1826], block_coordinates[1827], block_coordinates[1828], block_coordinates[1829], block_coordinates[1830], block_coordinates[1831], block_coordinates[1832], block_coordinates[1833], block_coordinates[1834], block_coordinates[1835], block_coordinates[1836], block_coordinates[1837], block_coordinates[1838], block_coordinates[1839], block_coordinates[1840], block_coordinates[1841], block_coordinates[1842], block_coordinates[1843], block_coordinates[1844], block_coordinates[1845], block_coordinates[1846], block_coordinates[1847], block_coordinates[1848], block_coordinates[1849], block_coordinates[1850], block_coordinates[1851], block_coordinates[1852], block_coordinates[1853], block_coordinates[1854], block_coordinates[1855], block_coordinates[1856], block_coordinates[1857], block_coordinates[1858], block_coordinates[1859], block_coordinates[1860], block_coordinates[1861], block_coordinates[1862], block_coordinates[1863], block_coordinates[1864], block_coordinates[1865], block_coordinates[1866], block_coordinates[1867], block_coordinates[1868], block_coordinates[1869], block_coordinates[1870], block_coordinates[1871], block_coordinates[1872], block_coordinates[1873], block_coordinates[1874], block_coordinates[1875], block_coordinates[1876], block_coordinates[1877], block_coordinates[1878], block_coordinates[1879], block_coordinates[1880], block_coordinates[1881], block_coordinates[1882], block_coordinates[1883], block_coordinates[1884], block_coordinates[1885], block_coordinates[1886], block_coordinates[1887], block_coordinates[1888], block_coordinates[1889], block_coordinates[1890], block_coordinates[1891], block_coordinates[1892], block_coordinates[1893], block_coordinates[1894], block_coordinates[1895], block_coordinates[1896], block_coordinates[1897], block_coordinates[1898], block_coordinates[1899], block_coordinates[1900], block_coordinates[1901], block_coordinates[1902], block_coordinates[1903], block_coordinates[1904], block_coordinates[1905], block_coordinates[1906], block_coordinates[1907], block_coordinates[1908], block_coordinates[1909], block_coordinates[1910], block_coordinates[1911], block_coordinates[1912], block_coordinates[1913], block_coordinates[1914], block_coordinates[1915], block_coordinates[1916], block_coordinates[1917], block_coordinates[1918], block_coordinates[1919], block_coordinates[1920], block_coordinates[1921], block_coordinates[1922], block_coordinates[1923], block_coordinates[1924], block_coordinates[1925], block_coordinates[1926], block_coordinates[1927], block_coordinates[1928], block_coordinates[1929], block_coordinates[1930], block_coordinates[1931], block_coordinates[1932], block_coordinates[1933], block_coordinates[1934], block_coordinates[1935], block_coordinates[1936], block_coordinates[1937], block_coordinates[1938], block_coordinates[1939], block_coordinates[1940], block_coordinates[1941], block_coordinates[1942], block_coordinates[1943], block_coordinates[1944], block_coordinates[1945], block_coordinates[1946], block_coordinates[1947], block_coordinates[1948], block_coordinates[1949], block_coordinates[1950], block_coordinates[1951], block_coordinates[1952], block_coordinates[1953], block_coordinates[1954], block_coordinates[1955], block_coordinates[1956], block_coordinates[1957], block_coordinates[1958], block_coordinates[1959], block_coordinates[1960], block_coordinates[1961], block_coordinates[1962], block_coordinates[1963], block_coordinates[1964], block_coordinates[1965], block_coordinates[1966], block_coordinates[1967], block_coordinates[1968], block_coordinates[1969], block_coordinates[1970], block_coordinates[1971], block_coordinates[1972], block_coordinates[1973], block_coordinates[1974], block_coordinates[1975], block_coordinates[1976], block_coordinates[1977], block_coordinates[1978], block_coordinates[1979], block_coordinates[1980], block_coordinates[1981], block_coordinates[1982], block_coordinates[1983], block_coordinates[1984], block_coordinates[1985], block_coordinates[1986], block_coordinates[1987], block_coordinates[1988], block_coordinates[1989], block_coordinates[1990], block_coordinates[1991], block_coordinates[1992], block_coordinates[1993], block_coordinates[1994], block_coordinates[1995], block_coordinates[1996], block_coordinates[1997], block_coordinates[1998], block_coordinates[1999], block_coordinates[2000], block_coordinates[2001], block_coordinates[2002], block_coordinates[2003], block_coordinates[2004], block_coordinates[2005], block_coordinates[2006], block_coordinates[2007], block_coordinates[2008], block_coordinates[2009], block_coordinates[2010], block_coordinates[2011], block_coordinates[2012], block_coordinates[2013], block_coordinates[2014], block_coordinates[2015], block_coordinates[2016], block_coordinates[2017], block_coordinates[2018], block_coordinates[2019], block_coordinates[2020], block_coordinates[2021], block_coordinates[2022], block_coordinates[2023], block_coordinates[2024], block_coordinates[2025], block_coordinates[2026], block_coordinates[2027], block_coordinates[2028], block_coordinates[2029], block_coordinates[2030], block_coordinates[2031], block_coordinates[2032], block_coordinates[2033], block_coordinates[2034], block_coordinates[2035], block_coordinates[2036], block_coordinates[2037], block_coordinates[2038], block_coordinates[2039], block_coordinates[2040], block_coordinates[2041], block_coordinates[2042], block_coordinates[2043], block_coordinates[2044], block_coordinates[2045], block_coordinates[2046], block_coordinates[2047], block_coordinates[2048], block_coordinates[2049], block_coordinates[2050], block_coordinates[2051], block_coordinates[2052], block_coordinates[2053], block_coordinates[2054], block_coordinates[2055], block_coordinates[2056], block_coordinates[2057], block_coordinates[2058], block_coordinates[2059], block_coordinates[2060], block_coordinates[2061], block_coordinates[2062], block_coordinates[2063], block_coordinates[2064], block_coordinates[2065], block_coordinates[2066], block_coordinates[2067], block_coordinates[2068], block_coordinates[2069], block_coordinates[2070], block_coordinates[2071], block_coordinates[2072], block_coordinates[2073], block_coordinates[2074], block_coordinates[2075], block_coordinates[2076], block_coordinates[2077], block_coordinates[2078], block_coordinates[2079], block_coordinates[2080], block_coordinates[2081], block_coordinates[2082], block_coordinates[2083], block_coordinates[2084], block_coordinates[2085], block_coordinates[2086], block_coordinates[2087], block_coordinates[2088], block_coordinates[2089], block_coordinates[2090], block_coordinates[2091], block_coordinates[2092], block_coordinates[2093], block_coordinates[2094], block_coordinates[2095], block_coordinates[2096], block_coordinates[2097], block_coordinates[2098], block_coordinates[2099], block_coordinates[2100], block_coordinates[2101], block_coordinates[2102], block_coordinates[2103], block_coordinates[2104], block_coordinates[2105], block_coordinates[2106], block_coordinates[2107], block_coordinates[2108], block_coordinates[2109], block_coordinates[2110], block_coordinates[2111], block_coordinates[2112], block_coordinates[2113], block_coordinates[2114], block_coordinates[2115], block_coordinates[2116], block_coordinates[2117], block_coordinates[2118], block_coordinates[2119], block_coordinates[2120], block_coordinates[2121], block_coordinates[2122], block_coordinates[2123], block_coordinates[2124], block_coordinates[2125], block_coordinates[2126], block_coordinates[2127], block_coordinates[2128], block_coordinates[2129], block_coordinates[2130], block_coordinates[2131], block_coordinates[2132], block_coordinates[2133], block_coordinates[2134], block_coordinates[2135], block_coordinates[2136], block_coordinates[2137], block_coordinates[2138], block_coordinates[2139], block_coordinates[2140], block_coordinates[2141], block_coordinates[2142], block_coordinates[2143], block_coordinates[2144], block_coordinates[2145], block_coordinates[2146], block_coordinates[2147], block_coordinates[2148], block_coordinates[2149], block_coordinates[2150], block_coordinates[2151], block_coordinates[2152], block_coordinates[2153], block_coordinates[2154], block_coordinates[2155], block_coordinates[2156], block_coordinates[2157], block_coordinates[2158], block_coordinates[2159], block_coordinates[2160], block_coordinates[2161], block_coordinates[2162], block_coordinates[2163], block_coordinates[2164], block_coordinates[2165], block_coordinates[2166], block_coordinates[2167], block_coordinates[2168], block_coordinates[2169], block_coordinates[2170], block_coordinates[2171], block_coordinates[2172], block_coordinates[2173], block_coordinates[2174], block_coordinates[2175], block_coordinates[2176], block_coordinates[2177], block_coordinates[2178], block_coordinates[2179], block_coordinates[2180], block_coordinates[2181], block_coordinates[2182], block_coordinates[2183], block_coordinates[2184], block_coordinates[2185], block_coordinates[2186]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t * block_direction_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_direction_streams[stream] = block_direction[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
        }
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_direction_streams, kappa, block_output_streams);

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

extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
