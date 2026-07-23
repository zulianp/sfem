#include <cstdio>
#include <type_traits>
#include "../neohookean_ogden_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
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
struct neohookean_ogden_proteus_hex729_affine_reference_data {
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
struct neohookean_ogden_proteus_hex729_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[90] = {scalar_t(0.74640942988869863), scalar_t(0.69587656250304208), scalar_t(-1.1507323407405976), scalar_t(1.5066574132709454), scalar_t(-1.399876561262831), scalar_t(0.8911457797574196), scalar_t(-0.36999603319584745), scalar_t(0.090382687697424524), scalar_t(-0.0098669379182538229), scalar_t(0.16655897497169198), scalar_t(1.5626107927930399), scalar_t(-1.7238032152037754), scalar_t(2.0462847873357397), scalar_t(-1.8186454006838009), scalar_t(1.1287204369742534), scalar_t(-0.4610023392379009), scalar_t(0.11132644226332365), scalar_t(-0.012050479212571269), scalar_t(-0.015802991866793473), scalar_t(0.57416143946571829), scalar_t(0.7906828209761454), scalar_t(-0.6607028536169085), scalar_t(0.52198287377648045), scalar_t(-0.30526060503638996), scalar_t(0.12027718562379788), scalar_t(-0.028354577146261196), scalar_t(0.0030167078242109044), scalar_t(0.002947726560881461), scalar_t(-0.042202682216816827), scalar_t(0.70213571394075658), scalar_t(0.50999615030512613), scalar_t(-0.26976216826647509), scalar_t(0.13686212373260215), scalar_t(-0.050102531884068956), scalar_t(0.011290870013716313), scalar_t(-0.0011652021857218211), scalar_t(-0.001380500832353452), scalar_t(0.01563705906658789), scalar_t(-0.093697028698593504), scalar_t(0.65066434782116778), scalar_t(0.55246981048409594), scalar_t(-0.16496138217989398), scalar_t(0.050702315010460089), scalar_t(-0.010457343212662354), scalar_t(0.0010227225411917102), scalar_t(0.0010227225411917102), scalar_t(-0.010457343212662354), scalar_t(0.050702315010460082), scalar_t(-0.16496138217989401), scalar_t(0.55246981048409594), scalar_t(0.65066434782116767), scalar_t(-0.093697028698593518), scalar_t(0.01563705906658789), scalar_t(-0.0013805008323534518), scalar_t(-0.0011652021857218209), scalar_t(0.011290870013716316), scalar_t(-0.050102531884068942), scalar_t(0.13686212373260212), scalar_t(-0.26976216826647514), scalar_t(0.50999615030512613), scalar_t(0.70213571394075658), scalar_t(-0.042202682216816827), scalar_t(0.0029477265608814614), scalar_t(0.0030167078242109048), scalar_t(-0.028354577146261196), scalar_t(0.12027718562379787), scalar_t(-0.30526060503638996), scalar_t(0.52198287377648045), scalar_t(-0.6607028536169085), scalar_t(0.79068282097614551), scalar_t(0.57416143946571829), scalar_t(-0.015802991866793473), scalar_t(-0.012050479212571279), scalar_t(0.11132644226332375), scalar_t(-0.46100233923790135), scalar_t(1.1287204369742541), scalar_t(-1.8186454006838022), scalar_t(2.046284787335741), scalar_t(-1.7238032152037763), scalar_t(1.5626107927930397), scalar_t(0.16655897497169225), scalar_t(-0.0098669379182538212), scalar_t(0.090382687697424538), scalar_t(-0.36999603319584751), scalar_t(0.89114577975741971), scalar_t(-1.3998765612628308), scalar_t(1.5066574132709454), scalar_t(-1.1507323407405976), scalar_t(0.69587656250304197), scalar_t(0.74640942988869863)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[90] = {scalar_t(-17.26694148249619), scalar_t(43.455045756406491), scalar_t(-66.436873931713464), scalar_t(84.790120860397977), scalar_t(-77.788021670259511), scalar_t(49.145173570116455), scalar_t(-20.302098604262468), scalar_t(4.9416140326286495), scalar_t(-0.53801853081792839), scalar_t(-5.6618957278934356), scalar_t(-2.7968240941219547), scalar_t(23.6041517514352), scalar_t(-32.576587362759803), scalar_t(30.661630189836799), scalar_t(-19.614844189747608), scalar_t(8.1627074098217314), scalar_t(-1.9964422225588745), scalar_t(0.21810424598794281), scalar_t(-0.049712140818957348), scalar_t(-10.879337285094028), scalar_t(16.234227869297992), scalar_t(-9.2774505852476459), scalar_t(6.4349861716914694), scalar_t(-3.521527715230778), scalar_t(1.3326706728866795), scalar_t(-0.30575905418678606), scalar_t(0.031902066702050903), scalar_t(0.037348400349857105), scalar_t(-0.41708952861996057), scalar_t(-9.7090736223749108), scalar_t(13.823668388366405), scalar_t(-5.6150351024265817), scalar_t(2.6177072228161569), scalar_t(-0.91901832284214957), scalar_t(0.20199467690044065), scalar_t(-0.020502112169261365), scalar_t(-0.0045612743167067083), scalar_t(0.036384465921104579), scalar_t(0.003941373810085802), scalar_t(-9.1896399633712367), scalar_t(10.545571395913317), scalar_t(-1.7598099842108774), scalar_t(0.44294365282621678), scalar_t(-0.082392429591471861), scalar_t(0.0075627630195701945), scalar_t(-0.0075627630195701885), scalar_t(0.082392429591471694), scalar_t(-0.44294365282621623), scalar_t(1.7598099842108761), scalar_t(-10.545571395913317), scalar_t(9.1896399633712385), scalar_t(-0.0039413738100856077), scalar_t(-0.036384465921104524), scalar_t(0.0045612743167067092), scalar_t(0.020502112169261361), scalar_t(-0.20199467690044065), scalar_t(0.91901832284214979), scalar_t(-2.6177072228161578), scalar_t(5.6150351024265799), scalar_t(-13.823668388366407), scalar_t(9.709073622374909), scalar_t(0.41708952861996063), scalar_t(-0.037348400349857105), scalar_t(-0.031902066702050882), scalar_t(0.30575905418678573), scalar_t(-1.3326706728866795), scalar_t(3.5215277152307771), scalar_t(-6.4349861716914747), scalar_t(9.2774505852476459), scalar_t(-16.234227869297996), scalar_t(10.879337285094028), scalar_t(0.049712140818957473), scalar_t(-0.21810424598794259), scalar_t(1.9964422225588709), scalar_t(-8.1627074098217243), scalar_t(19.61484418974759), scalar_t(-30.66163018983676), scalar_t(32.576587362759753), scalar_t(-23.604151751435154), scalar_t(2.7968240941219236), scalar_t(5.6618957278934428), scalar_t(0.53801853081792839), scalar_t(-4.9416140326286504), scalar_t(20.302098604262461), scalar_t(-49.145173570116434), scalar_t(77.788021670259482), scalar_t(-84.790120860397977), scalar_t(66.43687393171345), scalar_t(-43.455045756406484), scalar_t(17.26694148249619)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[10] = {scalar_t(0.03333567215434368), scalar_t(0.074725674575290224), scalar_t(0.10954318125799124), scalar_t(0.13463335965499829), scalar_t(0.14776211235737652), scalar_t(0.14776211235737652), scalar_t(0.13463335965499829), scalar_t(0.10954318125799124), scalar_t(0.074725674575290224), scalar_t(0.03333567215434368)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    19,
    21,
    0,
    0,
    10,
    0,
    1,
    0,
    6,
    5,
    70,
    453280,
    889560,
    4,
    12,
    10,
    180,
    10,
    2,
    2187,
    0,
    1,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl(
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
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
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

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl(
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
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
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

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_packed_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728], block_u_data[729], block_u_data[730], block_u_data[731], block_u_data[732], block_u_data[733], block_u_data[734], block_u_data[735], block_u_data[736], block_u_data[737], block_u_data[738], block_u_data[739], block_u_data[740], block_u_data[741], block_u_data[742], block_u_data[743], block_u_data[744], block_u_data[745], block_u_data[746], block_u_data[747], block_u_data[748], block_u_data[749], block_u_data[750], block_u_data[751], block_u_data[752], block_u_data[753], block_u_data[754], block_u_data[755], block_u_data[756], block_u_data[757], block_u_data[758], block_u_data[759], block_u_data[760], block_u_data[761], block_u_data[762], block_u_data[763], block_u_data[764], block_u_data[765], block_u_data[766], block_u_data[767], block_u_data[768], block_u_data[769], block_u_data[770], block_u_data[771], block_u_data[772], block_u_data[773], block_u_data[774], block_u_data[775], block_u_data[776], block_u_data[777], block_u_data[778], block_u_data[779], block_u_data[780], block_u_data[781], block_u_data[782], block_u_data[783], block_u_data[784], block_u_data[785], block_u_data[786], block_u_data[787], block_u_data[788], block_u_data[789], block_u_data[790], block_u_data[791], block_u_data[792], block_u_data[793], block_u_data[794], block_u_data[795], block_u_data[796], block_u_data[797], block_u_data[798], block_u_data[799], block_u_data[800], block_u_data[801], block_u_data[802], block_u_data[803], block_u_data[804], block_u_data[805], block_u_data[806], block_u_data[807], block_u_data[808], block_u_data[809], block_u_data[810], block_u_data[811], block_u_data[812], block_u_data[813], block_u_data[814], block_u_data[815], block_u_data[816], block_u_data[817], block_u_data[818], block_u_data[819], block_u_data[820], block_u_data[821], block_u_data[822], block_u_data[823], block_u_data[824], block_u_data[825], block_u_data[826], block_u_data[827], block_u_data[828], block_u_data[829], block_u_data[830], block_u_data[831], block_u_data[832], block_u_data[833], block_u_data[834], block_u_data[835], block_u_data[836], block_u_data[837], block_u_data[838], block_u_data[839], block_u_data[840], block_u_data[841], block_u_data[842], block_u_data[843], block_u_data[844], block_u_data[845], block_u_data[846], block_u_data[847], block_u_data[848], block_u_data[849], block_u_data[850], block_u_data[851], block_u_data[852], block_u_data[853], block_u_data[854], block_u_data[855], block_u_data[856], block_u_data[857], block_u_data[858], block_u_data[859], block_u_data[860], block_u_data[861], block_u_data[862], block_u_data[863], block_u_data[864], block_u_data[865], block_u_data[866], block_u_data[867], block_u_data[868], block_u_data[869], block_u_data[870], block_u_data[871], block_u_data[872], block_u_data[873], block_u_data[874], block_u_data[875], block_u_data[876], block_u_data[877], block_u_data[878], block_u_data[879], block_u_data[880], block_u_data[881], block_u_data[882], block_u_data[883], block_u_data[884], block_u_data[885], block_u_data[886], block_u_data[887], block_u_data[888], block_u_data[889], block_u_data[890], block_u_data[891], block_u_data[892], block_u_data[893], block_u_data[894], block_u_data[895], block_u_data[896], block_u_data[897], block_u_data[898], block_u_data[899], block_u_data[900], block_u_data[901], block_u_data[902], block_u_data[903], block_u_data[904], block_u_data[905], block_u_data[906], block_u_data[907], block_u_data[908], block_u_data[909], block_u_data[910], block_u_data[911], block_u_data[912], block_u_data[913], block_u_data[914], block_u_data[915], block_u_data[916], block_u_data[917], block_u_data[918], block_u_data[919], block_u_data[920], block_u_data[921], block_u_data[922], block_u_data[923], block_u_data[924], block_u_data[925], block_u_data[926], block_u_data[927], block_u_data[928], block_u_data[929], block_u_data[930], block_u_data[931], block_u_data[932], block_u_data[933], block_u_data[934], block_u_data[935], block_u_data[936], block_u_data[937], block_u_data[938], block_u_data[939], block_u_data[940], block_u_data[941], block_u_data[942], block_u_data[943], block_u_data[944], block_u_data[945], block_u_data[946], block_u_data[947], block_u_data[948], block_u_data[949], block_u_data[950], block_u_data[951], block_u_data[952], block_u_data[953], block_u_data[954], block_u_data[955], block_u_data[956], block_u_data[957], block_u_data[958], block_u_data[959], block_u_data[960], block_u_data[961], block_u_data[962], block_u_data[963], block_u_data[964], block_u_data[965], block_u_data[966], block_u_data[967], block_u_data[968], block_u_data[969], block_u_data[970], block_u_data[971], block_u_data[972], block_u_data[973], block_u_data[974], block_u_data[975], block_u_data[976], block_u_data[977], block_u_data[978], block_u_data[979], block_u_data[980], block_u_data[981], block_u_data[982], block_u_data[983], block_u_data[984], block_u_data[985], block_u_data[986], block_u_data[987], block_u_data[988], block_u_data[989], block_u_data[990], block_u_data[991], block_u_data[992], block_u_data[993], block_u_data[994], block_u_data[995], block_u_data[996], block_u_data[997], block_u_data[998], block_u_data[999], block_u_data[1000], block_u_data[1001], block_u_data[1002], block_u_data[1003], block_u_data[1004], block_u_data[1005], block_u_data[1006], block_u_data[1007], block_u_data[1008], block_u_data[1009], block_u_data[1010], block_u_data[1011], block_u_data[1012], block_u_data[1013], block_u_data[1014], block_u_data[1015], block_u_data[1016], block_u_data[1017], block_u_data[1018], block_u_data[1019], block_u_data[1020], block_u_data[1021], block_u_data[1022], block_u_data[1023], block_u_data[1024], block_u_data[1025], block_u_data[1026], block_u_data[1027], block_u_data[1028], block_u_data[1029], block_u_data[1030], block_u_data[1031], block_u_data[1032], block_u_data[1033], block_u_data[1034], block_u_data[1035], block_u_data[1036], block_u_data[1037], block_u_data[1038], block_u_data[1039], block_u_data[1040], block_u_data[1041], block_u_data[1042], block_u_data[1043], block_u_data[1044], block_u_data[1045], block_u_data[1046], block_u_data[1047], block_u_data[1048], block_u_data[1049], block_u_data[1050], block_u_data[1051], block_u_data[1052], block_u_data[1053], block_u_data[1054], block_u_data[1055], block_u_data[1056], block_u_data[1057], block_u_data[1058], block_u_data[1059], block_u_data[1060], block_u_data[1061], block_u_data[1062], block_u_data[1063], block_u_data[1064], block_u_data[1065], block_u_data[1066], block_u_data[1067], block_u_data[1068], block_u_data[1069], block_u_data[1070], block_u_data[1071], block_u_data[1072], block_u_data[1073], block_u_data[1074], block_u_data[1075], block_u_data[1076], block_u_data[1077], block_u_data[1078], block_u_data[1079], block_u_data[1080], block_u_data[1081], block_u_data[1082], block_u_data[1083], block_u_data[1084], block_u_data[1085], block_u_data[1086], block_u_data[1087], block_u_data[1088], block_u_data[1089], block_u_data[1090], block_u_data[1091], block_u_data[1092], block_u_data[1093], block_u_data[1094], block_u_data[1095], block_u_data[1096], block_u_data[1097], block_u_data[1098], block_u_data[1099], block_u_data[1100], block_u_data[1101], block_u_data[1102], block_u_data[1103], block_u_data[1104], block_u_data[1105], block_u_data[1106], block_u_data[1107], block_u_data[1108], block_u_data[1109], block_u_data[1110], block_u_data[1111], block_u_data[1112], block_u_data[1113], block_u_data[1114], block_u_data[1115], block_u_data[1116], block_u_data[1117], block_u_data[1118], block_u_data[1119], block_u_data[1120], block_u_data[1121], block_u_data[1122], block_u_data[1123], block_u_data[1124], block_u_data[1125], block_u_data[1126], block_u_data[1127], block_u_data[1128], block_u_data[1129], block_u_data[1130], block_u_data[1131], block_u_data[1132], block_u_data[1133], block_u_data[1134], block_u_data[1135], block_u_data[1136], block_u_data[1137], block_u_data[1138], block_u_data[1139], block_u_data[1140], block_u_data[1141], block_u_data[1142], block_u_data[1143], block_u_data[1144], block_u_data[1145], block_u_data[1146], block_u_data[1147], block_u_data[1148], block_u_data[1149], block_u_data[1150], block_u_data[1151], block_u_data[1152], block_u_data[1153], block_u_data[1154], block_u_data[1155], block_u_data[1156], block_u_data[1157], block_u_data[1158], block_u_data[1159], block_u_data[1160], block_u_data[1161], block_u_data[1162], block_u_data[1163], block_u_data[1164], block_u_data[1165], block_u_data[1166], block_u_data[1167], block_u_data[1168], block_u_data[1169], block_u_data[1170], block_u_data[1171], block_u_data[1172], block_u_data[1173], block_u_data[1174], block_u_data[1175], block_u_data[1176], block_u_data[1177], block_u_data[1178], block_u_data[1179], block_u_data[1180], block_u_data[1181], block_u_data[1182], block_u_data[1183], block_u_data[1184], block_u_data[1185], block_u_data[1186], block_u_data[1187], block_u_data[1188], block_u_data[1189], block_u_data[1190], block_u_data[1191], block_u_data[1192], block_u_data[1193], block_u_data[1194], block_u_data[1195], block_u_data[1196], block_u_data[1197], block_u_data[1198], block_u_data[1199], block_u_data[1200], block_u_data[1201], block_u_data[1202], block_u_data[1203], block_u_data[1204], block_u_data[1205], block_u_data[1206], block_u_data[1207], block_u_data[1208], block_u_data[1209], block_u_data[1210], block_u_data[1211], block_u_data[1212], block_u_data[1213], block_u_data[1214], block_u_data[1215], block_u_data[1216], block_u_data[1217], block_u_data[1218], block_u_data[1219], block_u_data[1220], block_u_data[1221], block_u_data[1222], block_u_data[1223], block_u_data[1224], block_u_data[1225], block_u_data[1226], block_u_data[1227], block_u_data[1228], block_u_data[1229], block_u_data[1230], block_u_data[1231], block_u_data[1232], block_u_data[1233], block_u_data[1234], block_u_data[1235], block_u_data[1236], block_u_data[1237], block_u_data[1238], block_u_data[1239], block_u_data[1240], block_u_data[1241], block_u_data[1242], block_u_data[1243], block_u_data[1244], block_u_data[1245], block_u_data[1246], block_u_data[1247], block_u_data[1248], block_u_data[1249], block_u_data[1250], block_u_data[1251], block_u_data[1252], block_u_data[1253], block_u_data[1254], block_u_data[1255], block_u_data[1256], block_u_data[1257], block_u_data[1258], block_u_data[1259], block_u_data[1260], block_u_data[1261], block_u_data[1262], block_u_data[1263], block_u_data[1264], block_u_data[1265], block_u_data[1266], block_u_data[1267], block_u_data[1268], block_u_data[1269], block_u_data[1270], block_u_data[1271], block_u_data[1272], block_u_data[1273], block_u_data[1274], block_u_data[1275], block_u_data[1276], block_u_data[1277], block_u_data[1278], block_u_data[1279], block_u_data[1280], block_u_data[1281], block_u_data[1282], block_u_data[1283], block_u_data[1284], block_u_data[1285], block_u_data[1286], block_u_data[1287], block_u_data[1288], block_u_data[1289], block_u_data[1290], block_u_data[1291], block_u_data[1292], block_u_data[1293], block_u_data[1294], block_u_data[1295], block_u_data[1296], block_u_data[1297], block_u_data[1298], block_u_data[1299], block_u_data[1300], block_u_data[1301], block_u_data[1302], block_u_data[1303], block_u_data[1304], block_u_data[1305], block_u_data[1306], block_u_data[1307], block_u_data[1308], block_u_data[1309], block_u_data[1310], block_u_data[1311], block_u_data[1312], block_u_data[1313], block_u_data[1314], block_u_data[1315], block_u_data[1316], block_u_data[1317], block_u_data[1318], block_u_data[1319], block_u_data[1320], block_u_data[1321], block_u_data[1322], block_u_data[1323], block_u_data[1324], block_u_data[1325], block_u_data[1326], block_u_data[1327], block_u_data[1328], block_u_data[1329], block_u_data[1330], block_u_data[1331], block_u_data[1332], block_u_data[1333], block_u_data[1334], block_u_data[1335], block_u_data[1336], block_u_data[1337], block_u_data[1338], block_u_data[1339], block_u_data[1340], block_u_data[1341], block_u_data[1342], block_u_data[1343], block_u_data[1344], block_u_data[1345], block_u_data[1346], block_u_data[1347], block_u_data[1348], block_u_data[1349], block_u_data[1350], block_u_data[1351], block_u_data[1352], block_u_data[1353], block_u_data[1354], block_u_data[1355], block_u_data[1356], block_u_data[1357], block_u_data[1358], block_u_data[1359], block_u_data[1360], block_u_data[1361], block_u_data[1362], block_u_data[1363], block_u_data[1364], block_u_data[1365], block_u_data[1366], block_u_data[1367], block_u_data[1368], block_u_data[1369], block_u_data[1370], block_u_data[1371], block_u_data[1372], block_u_data[1373], block_u_data[1374], block_u_data[1375], block_u_data[1376], block_u_data[1377], block_u_data[1378], block_u_data[1379], block_u_data[1380], block_u_data[1381], block_u_data[1382], block_u_data[1383], block_u_data[1384], block_u_data[1385], block_u_data[1386], block_u_data[1387], block_u_data[1388], block_u_data[1389], block_u_data[1390], block_u_data[1391], block_u_data[1392], block_u_data[1393], block_u_data[1394], block_u_data[1395], block_u_data[1396], block_u_data[1397], block_u_data[1398], block_u_data[1399], block_u_data[1400], block_u_data[1401], block_u_data[1402], block_u_data[1403], block_u_data[1404], block_u_data[1405], block_u_data[1406], block_u_data[1407], block_u_data[1408], block_u_data[1409], block_u_data[1410], block_u_data[1411], block_u_data[1412], block_u_data[1413], block_u_data[1414], block_u_data[1415], block_u_data[1416], block_u_data[1417], block_u_data[1418], block_u_data[1419], block_u_data[1420], block_u_data[1421], block_u_data[1422], block_u_data[1423], block_u_data[1424], block_u_data[1425], block_u_data[1426], block_u_data[1427], block_u_data[1428], block_u_data[1429], block_u_data[1430], block_u_data[1431], block_u_data[1432], block_u_data[1433], block_u_data[1434], block_u_data[1435], block_u_data[1436], block_u_data[1437], block_u_data[1438], block_u_data[1439], block_u_data[1440], block_u_data[1441], block_u_data[1442], block_u_data[1443], block_u_data[1444], block_u_data[1445], block_u_data[1446], block_u_data[1447], block_u_data[1448], block_u_data[1449], block_u_data[1450], block_u_data[1451], block_u_data[1452], block_u_data[1453], block_u_data[1454], block_u_data[1455], block_u_data[1456], block_u_data[1457], block_u_data[1458], block_u_data[1459], block_u_data[1460], block_u_data[1461], block_u_data[1462], block_u_data[1463], block_u_data[1464], block_u_data[1465], block_u_data[1466], block_u_data[1467], block_u_data[1468], block_u_data[1469], block_u_data[1470], block_u_data[1471], block_u_data[1472], block_u_data[1473], block_u_data[1474], block_u_data[1475], block_u_data[1476], block_u_data[1477], block_u_data[1478], block_u_data[1479], block_u_data[1480], block_u_data[1481], block_u_data[1482], block_u_data[1483], block_u_data[1484], block_u_data[1485], block_u_data[1486], block_u_data[1487], block_u_data[1488], block_u_data[1489], block_u_data[1490], block_u_data[1491], block_u_data[1492], block_u_data[1493], block_u_data[1494], block_u_data[1495], block_u_data[1496], block_u_data[1497], block_u_data[1498], block_u_data[1499], block_u_data[1500], block_u_data[1501], block_u_data[1502], block_u_data[1503], block_u_data[1504], block_u_data[1505], block_u_data[1506], block_u_data[1507], block_u_data[1508], block_u_data[1509], block_u_data[1510], block_u_data[1511], block_u_data[1512], block_u_data[1513], block_u_data[1514], block_u_data[1515], block_u_data[1516], block_u_data[1517], block_u_data[1518], block_u_data[1519], block_u_data[1520], block_u_data[1521], block_u_data[1522], block_u_data[1523], block_u_data[1524], block_u_data[1525], block_u_data[1526], block_u_data[1527], block_u_data[1528], block_u_data[1529], block_u_data[1530], block_u_data[1531], block_u_data[1532], block_u_data[1533], block_u_data[1534], block_u_data[1535], block_u_data[1536], block_u_data[1537], block_u_data[1538], block_u_data[1539], block_u_data[1540], block_u_data[1541], block_u_data[1542], block_u_data[1543], block_u_data[1544], block_u_data[1545], block_u_data[1546], block_u_data[1547], block_u_data[1548], block_u_data[1549], block_u_data[1550], block_u_data[1551], block_u_data[1552], block_u_data[1553], block_u_data[1554], block_u_data[1555], block_u_data[1556], block_u_data[1557], block_u_data[1558], block_u_data[1559], block_u_data[1560], block_u_data[1561], block_u_data[1562], block_u_data[1563], block_u_data[1564], block_u_data[1565], block_u_data[1566], block_u_data[1567], block_u_data[1568], block_u_data[1569], block_u_data[1570], block_u_data[1571], block_u_data[1572], block_u_data[1573], block_u_data[1574], block_u_data[1575], block_u_data[1576], block_u_data[1577], block_u_data[1578], block_u_data[1579], block_u_data[1580], block_u_data[1581], block_u_data[1582], block_u_data[1583], block_u_data[1584], block_u_data[1585], block_u_data[1586], block_u_data[1587], block_u_data[1588], block_u_data[1589], block_u_data[1590], block_u_data[1591], block_u_data[1592], block_u_data[1593], block_u_data[1594], block_u_data[1595], block_u_data[1596], block_u_data[1597], block_u_data[1598], block_u_data[1599], block_u_data[1600], block_u_data[1601], block_u_data[1602], block_u_data[1603], block_u_data[1604], block_u_data[1605], block_u_data[1606], block_u_data[1607], block_u_data[1608], block_u_data[1609], block_u_data[1610], block_u_data[1611], block_u_data[1612], block_u_data[1613], block_u_data[1614], block_u_data[1615], block_u_data[1616], block_u_data[1617], block_u_data[1618], block_u_data[1619], block_u_data[1620], block_u_data[1621], block_u_data[1622], block_u_data[1623], block_u_data[1624], block_u_data[1625], block_u_data[1626], block_u_data[1627], block_u_data[1628], block_u_data[1629], block_u_data[1630], block_u_data[1631], block_u_data[1632], block_u_data[1633], block_u_data[1634], block_u_data[1635], block_u_data[1636], block_u_data[1637], block_u_data[1638], block_u_data[1639], block_u_data[1640], block_u_data[1641], block_u_data[1642], block_u_data[1643], block_u_data[1644], block_u_data[1645], block_u_data[1646], block_u_data[1647], block_u_data[1648], block_u_data[1649], block_u_data[1650], block_u_data[1651], block_u_data[1652], block_u_data[1653], block_u_data[1654], block_u_data[1655], block_u_data[1656], block_u_data[1657], block_u_data[1658], block_u_data[1659], block_u_data[1660], block_u_data[1661], block_u_data[1662], block_u_data[1663], block_u_data[1664], block_u_data[1665], block_u_data[1666], block_u_data[1667], block_u_data[1668], block_u_data[1669], block_u_data[1670], block_u_data[1671], block_u_data[1672], block_u_data[1673], block_u_data[1674], block_u_data[1675], block_u_data[1676], block_u_data[1677], block_u_data[1678], block_u_data[1679], block_u_data[1680], block_u_data[1681], block_u_data[1682], block_u_data[1683], block_u_data[1684], block_u_data[1685], block_u_data[1686], block_u_data[1687], block_u_data[1688], block_u_data[1689], block_u_data[1690], block_u_data[1691], block_u_data[1692], block_u_data[1693], block_u_data[1694], block_u_data[1695], block_u_data[1696], block_u_data[1697], block_u_data[1698], block_u_data[1699], block_u_data[1700], block_u_data[1701], block_u_data[1702], block_u_data[1703], block_u_data[1704], block_u_data[1705], block_u_data[1706], block_u_data[1707], block_u_data[1708], block_u_data[1709], block_u_data[1710], block_u_data[1711], block_u_data[1712], block_u_data[1713], block_u_data[1714], block_u_data[1715], block_u_data[1716], block_u_data[1717], block_u_data[1718], block_u_data[1719], block_u_data[1720], block_u_data[1721], block_u_data[1722], block_u_data[1723], block_u_data[1724], block_u_data[1725], block_u_data[1726], block_u_data[1727], block_u_data[1728], block_u_data[1729], block_u_data[1730], block_u_data[1731], block_u_data[1732], block_u_data[1733], block_u_data[1734], block_u_data[1735], block_u_data[1736], block_u_data[1737], block_u_data[1738], block_u_data[1739], block_u_data[1740], block_u_data[1741], block_u_data[1742], block_u_data[1743], block_u_data[1744], block_u_data[1745], block_u_data[1746], block_u_data[1747], block_u_data[1748], block_u_data[1749], block_u_data[1750], block_u_data[1751], block_u_data[1752], block_u_data[1753], block_u_data[1754], block_u_data[1755], block_u_data[1756], block_u_data[1757], block_u_data[1758], block_u_data[1759], block_u_data[1760], block_u_data[1761], block_u_data[1762], block_u_data[1763], block_u_data[1764], block_u_data[1765], block_u_data[1766], block_u_data[1767], block_u_data[1768], block_u_data[1769], block_u_data[1770], block_u_data[1771], block_u_data[1772], block_u_data[1773], block_u_data[1774], block_u_data[1775], block_u_data[1776], block_u_data[1777], block_u_data[1778], block_u_data[1779], block_u_data[1780], block_u_data[1781], block_u_data[1782], block_u_data[1783], block_u_data[1784], block_u_data[1785], block_u_data[1786], block_u_data[1787], block_u_data[1788], block_u_data[1789], block_u_data[1790], block_u_data[1791], block_u_data[1792], block_u_data[1793], block_u_data[1794], block_u_data[1795], block_u_data[1796], block_u_data[1797], block_u_data[1798], block_u_data[1799], block_u_data[1800], block_u_data[1801], block_u_data[1802], block_u_data[1803], block_u_data[1804], block_u_data[1805], block_u_data[1806], block_u_data[1807], block_u_data[1808], block_u_data[1809], block_u_data[1810], block_u_data[1811], block_u_data[1812], block_u_data[1813], block_u_data[1814], block_u_data[1815], block_u_data[1816], block_u_data[1817], block_u_data[1818], block_u_data[1819], block_u_data[1820], block_u_data[1821], block_u_data[1822], block_u_data[1823], block_u_data[1824], block_u_data[1825], block_u_data[1826], block_u_data[1827], block_u_data[1828], block_u_data[1829], block_u_data[1830], block_u_data[1831], block_u_data[1832], block_u_data[1833], block_u_data[1834], block_u_data[1835], block_u_data[1836], block_u_data[1837], block_u_data[1838], block_u_data[1839], block_u_data[1840], block_u_data[1841], block_u_data[1842], block_u_data[1843], block_u_data[1844], block_u_data[1845], block_u_data[1846], block_u_data[1847], block_u_data[1848], block_u_data[1849], block_u_data[1850], block_u_data[1851], block_u_data[1852], block_u_data[1853], block_u_data[1854], block_u_data[1855], block_u_data[1856], block_u_data[1857], block_u_data[1858], block_u_data[1859], block_u_data[1860], block_u_data[1861], block_u_data[1862], block_u_data[1863], block_u_data[1864], block_u_data[1865], block_u_data[1866], block_u_data[1867], block_u_data[1868], block_u_data[1869], block_u_data[1870], block_u_data[1871], block_u_data[1872], block_u_data[1873], block_u_data[1874], block_u_data[1875], block_u_data[1876], block_u_data[1877], block_u_data[1878], block_u_data[1879], block_u_data[1880], block_u_data[1881], block_u_data[1882], block_u_data[1883], block_u_data[1884], block_u_data[1885], block_u_data[1886], block_u_data[1887], block_u_data[1888], block_u_data[1889], block_u_data[1890], block_u_data[1891], block_u_data[1892], block_u_data[1893], block_u_data[1894], block_u_data[1895], block_u_data[1896], block_u_data[1897], block_u_data[1898], block_u_data[1899], block_u_data[1900], block_u_data[1901], block_u_data[1902], block_u_data[1903], block_u_data[1904], block_u_data[1905], block_u_data[1906], block_u_data[1907], block_u_data[1908], block_u_data[1909], block_u_data[1910], block_u_data[1911], block_u_data[1912], block_u_data[1913], block_u_data[1914], block_u_data[1915], block_u_data[1916], block_u_data[1917], block_u_data[1918], block_u_data[1919], block_u_data[1920], block_u_data[1921], block_u_data[1922], block_u_data[1923], block_u_data[1924], block_u_data[1925], block_u_data[1926], block_u_data[1927], block_u_data[1928], block_u_data[1929], block_u_data[1930], block_u_data[1931], block_u_data[1932], block_u_data[1933], block_u_data[1934], block_u_data[1935], block_u_data[1936], block_u_data[1937], block_u_data[1938], block_u_data[1939], block_u_data[1940], block_u_data[1941], block_u_data[1942], block_u_data[1943], block_u_data[1944], block_u_data[1945], block_u_data[1946], block_u_data[1947], block_u_data[1948], block_u_data[1949], block_u_data[1950], block_u_data[1951], block_u_data[1952], block_u_data[1953], block_u_data[1954], block_u_data[1955], block_u_data[1956], block_u_data[1957], block_u_data[1958], block_u_data[1959], block_u_data[1960], block_u_data[1961], block_u_data[1962], block_u_data[1963], block_u_data[1964], block_u_data[1965], block_u_data[1966], block_u_data[1967], block_u_data[1968], block_u_data[1969], block_u_data[1970], block_u_data[1971], block_u_data[1972], block_u_data[1973], block_u_data[1974], block_u_data[1975], block_u_data[1976], block_u_data[1977], block_u_data[1978], block_u_data[1979], block_u_data[1980], block_u_data[1981], block_u_data[1982], block_u_data[1983], block_u_data[1984], block_u_data[1985], block_u_data[1986], block_u_data[1987], block_u_data[1988], block_u_data[1989], block_u_data[1990], block_u_data[1991], block_u_data[1992], block_u_data[1993], block_u_data[1994], block_u_data[1995], block_u_data[1996], block_u_data[1997], block_u_data[1998], block_u_data[1999], block_u_data[2000], block_u_data[2001], block_u_data[2002], block_u_data[2003], block_u_data[2004], block_u_data[2005], block_u_data[2006], block_u_data[2007], block_u_data[2008], block_u_data[2009], block_u_data[2010], block_u_data[2011], block_u_data[2012], block_u_data[2013], block_u_data[2014], block_u_data[2015], block_u_data[2016], block_u_data[2017], block_u_data[2018], block_u_data[2019], block_u_data[2020], block_u_data[2021], block_u_data[2022], block_u_data[2023], block_u_data[2024], block_u_data[2025], block_u_data[2026], block_u_data[2027], block_u_data[2028], block_u_data[2029], block_u_data[2030], block_u_data[2031], block_u_data[2032], block_u_data[2033], block_u_data[2034], block_u_data[2035], block_u_data[2036], block_u_data[2037], block_u_data[2038], block_u_data[2039], block_u_data[2040], block_u_data[2041], block_u_data[2042], block_u_data[2043], block_u_data[2044], block_u_data[2045], block_u_data[2046], block_u_data[2047], block_u_data[2048], block_u_data[2049], block_u_data[2050], block_u_data[2051], block_u_data[2052], block_u_data[2053], block_u_data[2054], block_u_data[2055], block_u_data[2056], block_u_data[2057], block_u_data[2058], block_u_data[2059], block_u_data[2060], block_u_data[2061], block_u_data[2062], block_u_data[2063], block_u_data[2064], block_u_data[2065], block_u_data[2066], block_u_data[2067], block_u_data[2068], block_u_data[2069], block_u_data[2070], block_u_data[2071], block_u_data[2072], block_u_data[2073], block_u_data[2074], block_u_data[2075], block_u_data[2076], block_u_data[2077], block_u_data[2078], block_u_data[2079], block_u_data[2080], block_u_data[2081], block_u_data[2082], block_u_data[2083], block_u_data[2084], block_u_data[2085], block_u_data[2086], block_u_data[2087], block_u_data[2088], block_u_data[2089], block_u_data[2090], block_u_data[2091], block_u_data[2092], block_u_data[2093], block_u_data[2094], block_u_data[2095], block_u_data[2096], block_u_data[2097], block_u_data[2098], block_u_data[2099], block_u_data[2100], block_u_data[2101], block_u_data[2102], block_u_data[2103], block_u_data[2104], block_u_data[2105], block_u_data[2106], block_u_data[2107], block_u_data[2108], block_u_data[2109], block_u_data[2110], block_u_data[2111], block_u_data[2112], block_u_data[2113], block_u_data[2114], block_u_data[2115], block_u_data[2116], block_u_data[2117], block_u_data[2118], block_u_data[2119], block_u_data[2120], block_u_data[2121], block_u_data[2122], block_u_data[2123], block_u_data[2124], block_u_data[2125], block_u_data[2126], block_u_data[2127], block_u_data[2128], block_u_data[2129], block_u_data[2130], block_u_data[2131], block_u_data[2132], block_u_data[2133], block_u_data[2134], block_u_data[2135], block_u_data[2136], block_u_data[2137], block_u_data[2138], block_u_data[2139], block_u_data[2140], block_u_data[2141], block_u_data[2142], block_u_data[2143], block_u_data[2144], block_u_data[2145], block_u_data[2146], block_u_data[2147], block_u_data[2148], block_u_data[2149], block_u_data[2150], block_u_data[2151], block_u_data[2152], block_u_data[2153], block_u_data[2154], block_u_data[2155], block_u_data[2156], block_u_data[2157], block_u_data[2158], block_u_data[2159], block_u_data[2160], block_u_data[2161], block_u_data[2162], block_u_data[2163], block_u_data[2164], block_u_data[2165], block_u_data[2166], block_u_data[2167], block_u_data[2168], block_u_data[2169], block_u_data[2170], block_u_data[2171], block_u_data[2172], block_u_data[2173], block_u_data[2174], block_u_data[2175], block_u_data[2176], block_u_data[2177], block_u_data[2178], block_u_data[2179], block_u_data[2180], block_u_data[2181], block_u_data[2182], block_u_data[2183], block_u_data[2184], block_u_data[2185], block_u_data[2186]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
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

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_packed_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728], block_u_data[729], block_u_data[730], block_u_data[731], block_u_data[732], block_u_data[733], block_u_data[734], block_u_data[735], block_u_data[736], block_u_data[737], block_u_data[738], block_u_data[739], block_u_data[740], block_u_data[741], block_u_data[742], block_u_data[743], block_u_data[744], block_u_data[745], block_u_data[746], block_u_data[747], block_u_data[748], block_u_data[749], block_u_data[750], block_u_data[751], block_u_data[752], block_u_data[753], block_u_data[754], block_u_data[755], block_u_data[756], block_u_data[757], block_u_data[758], block_u_data[759], block_u_data[760], block_u_data[761], block_u_data[762], block_u_data[763], block_u_data[764], block_u_data[765], block_u_data[766], block_u_data[767], block_u_data[768], block_u_data[769], block_u_data[770], block_u_data[771], block_u_data[772], block_u_data[773], block_u_data[774], block_u_data[775], block_u_data[776], block_u_data[777], block_u_data[778], block_u_data[779], block_u_data[780], block_u_data[781], block_u_data[782], block_u_data[783], block_u_data[784], block_u_data[785], block_u_data[786], block_u_data[787], block_u_data[788], block_u_data[789], block_u_data[790], block_u_data[791], block_u_data[792], block_u_data[793], block_u_data[794], block_u_data[795], block_u_data[796], block_u_data[797], block_u_data[798], block_u_data[799], block_u_data[800], block_u_data[801], block_u_data[802], block_u_data[803], block_u_data[804], block_u_data[805], block_u_data[806], block_u_data[807], block_u_data[808], block_u_data[809], block_u_data[810], block_u_data[811], block_u_data[812], block_u_data[813], block_u_data[814], block_u_data[815], block_u_data[816], block_u_data[817], block_u_data[818], block_u_data[819], block_u_data[820], block_u_data[821], block_u_data[822], block_u_data[823], block_u_data[824], block_u_data[825], block_u_data[826], block_u_data[827], block_u_data[828], block_u_data[829], block_u_data[830], block_u_data[831], block_u_data[832], block_u_data[833], block_u_data[834], block_u_data[835], block_u_data[836], block_u_data[837], block_u_data[838], block_u_data[839], block_u_data[840], block_u_data[841], block_u_data[842], block_u_data[843], block_u_data[844], block_u_data[845], block_u_data[846], block_u_data[847], block_u_data[848], block_u_data[849], block_u_data[850], block_u_data[851], block_u_data[852], block_u_data[853], block_u_data[854], block_u_data[855], block_u_data[856], block_u_data[857], block_u_data[858], block_u_data[859], block_u_data[860], block_u_data[861], block_u_data[862], block_u_data[863], block_u_data[864], block_u_data[865], block_u_data[866], block_u_data[867], block_u_data[868], block_u_data[869], block_u_data[870], block_u_data[871], block_u_data[872], block_u_data[873], block_u_data[874], block_u_data[875], block_u_data[876], block_u_data[877], block_u_data[878], block_u_data[879], block_u_data[880], block_u_data[881], block_u_data[882], block_u_data[883], block_u_data[884], block_u_data[885], block_u_data[886], block_u_data[887], block_u_data[888], block_u_data[889], block_u_data[890], block_u_data[891], block_u_data[892], block_u_data[893], block_u_data[894], block_u_data[895], block_u_data[896], block_u_data[897], block_u_data[898], block_u_data[899], block_u_data[900], block_u_data[901], block_u_data[902], block_u_data[903], block_u_data[904], block_u_data[905], block_u_data[906], block_u_data[907], block_u_data[908], block_u_data[909], block_u_data[910], block_u_data[911], block_u_data[912], block_u_data[913], block_u_data[914], block_u_data[915], block_u_data[916], block_u_data[917], block_u_data[918], block_u_data[919], block_u_data[920], block_u_data[921], block_u_data[922], block_u_data[923], block_u_data[924], block_u_data[925], block_u_data[926], block_u_data[927], block_u_data[928], block_u_data[929], block_u_data[930], block_u_data[931], block_u_data[932], block_u_data[933], block_u_data[934], block_u_data[935], block_u_data[936], block_u_data[937], block_u_data[938], block_u_data[939], block_u_data[940], block_u_data[941], block_u_data[942], block_u_data[943], block_u_data[944], block_u_data[945], block_u_data[946], block_u_data[947], block_u_data[948], block_u_data[949], block_u_data[950], block_u_data[951], block_u_data[952], block_u_data[953], block_u_data[954], block_u_data[955], block_u_data[956], block_u_data[957], block_u_data[958], block_u_data[959], block_u_data[960], block_u_data[961], block_u_data[962], block_u_data[963], block_u_data[964], block_u_data[965], block_u_data[966], block_u_data[967], block_u_data[968], block_u_data[969], block_u_data[970], block_u_data[971], block_u_data[972], block_u_data[973], block_u_data[974], block_u_data[975], block_u_data[976], block_u_data[977], block_u_data[978], block_u_data[979], block_u_data[980], block_u_data[981], block_u_data[982], block_u_data[983], block_u_data[984], block_u_data[985], block_u_data[986], block_u_data[987], block_u_data[988], block_u_data[989], block_u_data[990], block_u_data[991], block_u_data[992], block_u_data[993], block_u_data[994], block_u_data[995], block_u_data[996], block_u_data[997], block_u_data[998], block_u_data[999], block_u_data[1000], block_u_data[1001], block_u_data[1002], block_u_data[1003], block_u_data[1004], block_u_data[1005], block_u_data[1006], block_u_data[1007], block_u_data[1008], block_u_data[1009], block_u_data[1010], block_u_data[1011], block_u_data[1012], block_u_data[1013], block_u_data[1014], block_u_data[1015], block_u_data[1016], block_u_data[1017], block_u_data[1018], block_u_data[1019], block_u_data[1020], block_u_data[1021], block_u_data[1022], block_u_data[1023], block_u_data[1024], block_u_data[1025], block_u_data[1026], block_u_data[1027], block_u_data[1028], block_u_data[1029], block_u_data[1030], block_u_data[1031], block_u_data[1032], block_u_data[1033], block_u_data[1034], block_u_data[1035], block_u_data[1036], block_u_data[1037], block_u_data[1038], block_u_data[1039], block_u_data[1040], block_u_data[1041], block_u_data[1042], block_u_data[1043], block_u_data[1044], block_u_data[1045], block_u_data[1046], block_u_data[1047], block_u_data[1048], block_u_data[1049], block_u_data[1050], block_u_data[1051], block_u_data[1052], block_u_data[1053], block_u_data[1054], block_u_data[1055], block_u_data[1056], block_u_data[1057], block_u_data[1058], block_u_data[1059], block_u_data[1060], block_u_data[1061], block_u_data[1062], block_u_data[1063], block_u_data[1064], block_u_data[1065], block_u_data[1066], block_u_data[1067], block_u_data[1068], block_u_data[1069], block_u_data[1070], block_u_data[1071], block_u_data[1072], block_u_data[1073], block_u_data[1074], block_u_data[1075], block_u_data[1076], block_u_data[1077], block_u_data[1078], block_u_data[1079], block_u_data[1080], block_u_data[1081], block_u_data[1082], block_u_data[1083], block_u_data[1084], block_u_data[1085], block_u_data[1086], block_u_data[1087], block_u_data[1088], block_u_data[1089], block_u_data[1090], block_u_data[1091], block_u_data[1092], block_u_data[1093], block_u_data[1094], block_u_data[1095], block_u_data[1096], block_u_data[1097], block_u_data[1098], block_u_data[1099], block_u_data[1100], block_u_data[1101], block_u_data[1102], block_u_data[1103], block_u_data[1104], block_u_data[1105], block_u_data[1106], block_u_data[1107], block_u_data[1108], block_u_data[1109], block_u_data[1110], block_u_data[1111], block_u_data[1112], block_u_data[1113], block_u_data[1114], block_u_data[1115], block_u_data[1116], block_u_data[1117], block_u_data[1118], block_u_data[1119], block_u_data[1120], block_u_data[1121], block_u_data[1122], block_u_data[1123], block_u_data[1124], block_u_data[1125], block_u_data[1126], block_u_data[1127], block_u_data[1128], block_u_data[1129], block_u_data[1130], block_u_data[1131], block_u_data[1132], block_u_data[1133], block_u_data[1134], block_u_data[1135], block_u_data[1136], block_u_data[1137], block_u_data[1138], block_u_data[1139], block_u_data[1140], block_u_data[1141], block_u_data[1142], block_u_data[1143], block_u_data[1144], block_u_data[1145], block_u_data[1146], block_u_data[1147], block_u_data[1148], block_u_data[1149], block_u_data[1150], block_u_data[1151], block_u_data[1152], block_u_data[1153], block_u_data[1154], block_u_data[1155], block_u_data[1156], block_u_data[1157], block_u_data[1158], block_u_data[1159], block_u_data[1160], block_u_data[1161], block_u_data[1162], block_u_data[1163], block_u_data[1164], block_u_data[1165], block_u_data[1166], block_u_data[1167], block_u_data[1168], block_u_data[1169], block_u_data[1170], block_u_data[1171], block_u_data[1172], block_u_data[1173], block_u_data[1174], block_u_data[1175], block_u_data[1176], block_u_data[1177], block_u_data[1178], block_u_data[1179], block_u_data[1180], block_u_data[1181], block_u_data[1182], block_u_data[1183], block_u_data[1184], block_u_data[1185], block_u_data[1186], block_u_data[1187], block_u_data[1188], block_u_data[1189], block_u_data[1190], block_u_data[1191], block_u_data[1192], block_u_data[1193], block_u_data[1194], block_u_data[1195], block_u_data[1196], block_u_data[1197], block_u_data[1198], block_u_data[1199], block_u_data[1200], block_u_data[1201], block_u_data[1202], block_u_data[1203], block_u_data[1204], block_u_data[1205], block_u_data[1206], block_u_data[1207], block_u_data[1208], block_u_data[1209], block_u_data[1210], block_u_data[1211], block_u_data[1212], block_u_data[1213], block_u_data[1214], block_u_data[1215], block_u_data[1216], block_u_data[1217], block_u_data[1218], block_u_data[1219], block_u_data[1220], block_u_data[1221], block_u_data[1222], block_u_data[1223], block_u_data[1224], block_u_data[1225], block_u_data[1226], block_u_data[1227], block_u_data[1228], block_u_data[1229], block_u_data[1230], block_u_data[1231], block_u_data[1232], block_u_data[1233], block_u_data[1234], block_u_data[1235], block_u_data[1236], block_u_data[1237], block_u_data[1238], block_u_data[1239], block_u_data[1240], block_u_data[1241], block_u_data[1242], block_u_data[1243], block_u_data[1244], block_u_data[1245], block_u_data[1246], block_u_data[1247], block_u_data[1248], block_u_data[1249], block_u_data[1250], block_u_data[1251], block_u_data[1252], block_u_data[1253], block_u_data[1254], block_u_data[1255], block_u_data[1256], block_u_data[1257], block_u_data[1258], block_u_data[1259], block_u_data[1260], block_u_data[1261], block_u_data[1262], block_u_data[1263], block_u_data[1264], block_u_data[1265], block_u_data[1266], block_u_data[1267], block_u_data[1268], block_u_data[1269], block_u_data[1270], block_u_data[1271], block_u_data[1272], block_u_data[1273], block_u_data[1274], block_u_data[1275], block_u_data[1276], block_u_data[1277], block_u_data[1278], block_u_data[1279], block_u_data[1280], block_u_data[1281], block_u_data[1282], block_u_data[1283], block_u_data[1284], block_u_data[1285], block_u_data[1286], block_u_data[1287], block_u_data[1288], block_u_data[1289], block_u_data[1290], block_u_data[1291], block_u_data[1292], block_u_data[1293], block_u_data[1294], block_u_data[1295], block_u_data[1296], block_u_data[1297], block_u_data[1298], block_u_data[1299], block_u_data[1300], block_u_data[1301], block_u_data[1302], block_u_data[1303], block_u_data[1304], block_u_data[1305], block_u_data[1306], block_u_data[1307], block_u_data[1308], block_u_data[1309], block_u_data[1310], block_u_data[1311], block_u_data[1312], block_u_data[1313], block_u_data[1314], block_u_data[1315], block_u_data[1316], block_u_data[1317], block_u_data[1318], block_u_data[1319], block_u_data[1320], block_u_data[1321], block_u_data[1322], block_u_data[1323], block_u_data[1324], block_u_data[1325], block_u_data[1326], block_u_data[1327], block_u_data[1328], block_u_data[1329], block_u_data[1330], block_u_data[1331], block_u_data[1332], block_u_data[1333], block_u_data[1334], block_u_data[1335], block_u_data[1336], block_u_data[1337], block_u_data[1338], block_u_data[1339], block_u_data[1340], block_u_data[1341], block_u_data[1342], block_u_data[1343], block_u_data[1344], block_u_data[1345], block_u_data[1346], block_u_data[1347], block_u_data[1348], block_u_data[1349], block_u_data[1350], block_u_data[1351], block_u_data[1352], block_u_data[1353], block_u_data[1354], block_u_data[1355], block_u_data[1356], block_u_data[1357], block_u_data[1358], block_u_data[1359], block_u_data[1360], block_u_data[1361], block_u_data[1362], block_u_data[1363], block_u_data[1364], block_u_data[1365], block_u_data[1366], block_u_data[1367], block_u_data[1368], block_u_data[1369], block_u_data[1370], block_u_data[1371], block_u_data[1372], block_u_data[1373], block_u_data[1374], block_u_data[1375], block_u_data[1376], block_u_data[1377], block_u_data[1378], block_u_data[1379], block_u_data[1380], block_u_data[1381], block_u_data[1382], block_u_data[1383], block_u_data[1384], block_u_data[1385], block_u_data[1386], block_u_data[1387], block_u_data[1388], block_u_data[1389], block_u_data[1390], block_u_data[1391], block_u_data[1392], block_u_data[1393], block_u_data[1394], block_u_data[1395], block_u_data[1396], block_u_data[1397], block_u_data[1398], block_u_data[1399], block_u_data[1400], block_u_data[1401], block_u_data[1402], block_u_data[1403], block_u_data[1404], block_u_data[1405], block_u_data[1406], block_u_data[1407], block_u_data[1408], block_u_data[1409], block_u_data[1410], block_u_data[1411], block_u_data[1412], block_u_data[1413], block_u_data[1414], block_u_data[1415], block_u_data[1416], block_u_data[1417], block_u_data[1418], block_u_data[1419], block_u_data[1420], block_u_data[1421], block_u_data[1422], block_u_data[1423], block_u_data[1424], block_u_data[1425], block_u_data[1426], block_u_data[1427], block_u_data[1428], block_u_data[1429], block_u_data[1430], block_u_data[1431], block_u_data[1432], block_u_data[1433], block_u_data[1434], block_u_data[1435], block_u_data[1436], block_u_data[1437], block_u_data[1438], block_u_data[1439], block_u_data[1440], block_u_data[1441], block_u_data[1442], block_u_data[1443], block_u_data[1444], block_u_data[1445], block_u_data[1446], block_u_data[1447], block_u_data[1448], block_u_data[1449], block_u_data[1450], block_u_data[1451], block_u_data[1452], block_u_data[1453], block_u_data[1454], block_u_data[1455], block_u_data[1456], block_u_data[1457], block_u_data[1458], block_u_data[1459], block_u_data[1460], block_u_data[1461], block_u_data[1462], block_u_data[1463], block_u_data[1464], block_u_data[1465], block_u_data[1466], block_u_data[1467], block_u_data[1468], block_u_data[1469], block_u_data[1470], block_u_data[1471], block_u_data[1472], block_u_data[1473], block_u_data[1474], block_u_data[1475], block_u_data[1476], block_u_data[1477], block_u_data[1478], block_u_data[1479], block_u_data[1480], block_u_data[1481], block_u_data[1482], block_u_data[1483], block_u_data[1484], block_u_data[1485], block_u_data[1486], block_u_data[1487], block_u_data[1488], block_u_data[1489], block_u_data[1490], block_u_data[1491], block_u_data[1492], block_u_data[1493], block_u_data[1494], block_u_data[1495], block_u_data[1496], block_u_data[1497], block_u_data[1498], block_u_data[1499], block_u_data[1500], block_u_data[1501], block_u_data[1502], block_u_data[1503], block_u_data[1504], block_u_data[1505], block_u_data[1506], block_u_data[1507], block_u_data[1508], block_u_data[1509], block_u_data[1510], block_u_data[1511], block_u_data[1512], block_u_data[1513], block_u_data[1514], block_u_data[1515], block_u_data[1516], block_u_data[1517], block_u_data[1518], block_u_data[1519], block_u_data[1520], block_u_data[1521], block_u_data[1522], block_u_data[1523], block_u_data[1524], block_u_data[1525], block_u_data[1526], block_u_data[1527], block_u_data[1528], block_u_data[1529], block_u_data[1530], block_u_data[1531], block_u_data[1532], block_u_data[1533], block_u_data[1534], block_u_data[1535], block_u_data[1536], block_u_data[1537], block_u_data[1538], block_u_data[1539], block_u_data[1540], block_u_data[1541], block_u_data[1542], block_u_data[1543], block_u_data[1544], block_u_data[1545], block_u_data[1546], block_u_data[1547], block_u_data[1548], block_u_data[1549], block_u_data[1550], block_u_data[1551], block_u_data[1552], block_u_data[1553], block_u_data[1554], block_u_data[1555], block_u_data[1556], block_u_data[1557], block_u_data[1558], block_u_data[1559], block_u_data[1560], block_u_data[1561], block_u_data[1562], block_u_data[1563], block_u_data[1564], block_u_data[1565], block_u_data[1566], block_u_data[1567], block_u_data[1568], block_u_data[1569], block_u_data[1570], block_u_data[1571], block_u_data[1572], block_u_data[1573], block_u_data[1574], block_u_data[1575], block_u_data[1576], block_u_data[1577], block_u_data[1578], block_u_data[1579], block_u_data[1580], block_u_data[1581], block_u_data[1582], block_u_data[1583], block_u_data[1584], block_u_data[1585], block_u_data[1586], block_u_data[1587], block_u_data[1588], block_u_data[1589], block_u_data[1590], block_u_data[1591], block_u_data[1592], block_u_data[1593], block_u_data[1594], block_u_data[1595], block_u_data[1596], block_u_data[1597], block_u_data[1598], block_u_data[1599], block_u_data[1600], block_u_data[1601], block_u_data[1602], block_u_data[1603], block_u_data[1604], block_u_data[1605], block_u_data[1606], block_u_data[1607], block_u_data[1608], block_u_data[1609], block_u_data[1610], block_u_data[1611], block_u_data[1612], block_u_data[1613], block_u_data[1614], block_u_data[1615], block_u_data[1616], block_u_data[1617], block_u_data[1618], block_u_data[1619], block_u_data[1620], block_u_data[1621], block_u_data[1622], block_u_data[1623], block_u_data[1624], block_u_data[1625], block_u_data[1626], block_u_data[1627], block_u_data[1628], block_u_data[1629], block_u_data[1630], block_u_data[1631], block_u_data[1632], block_u_data[1633], block_u_data[1634], block_u_data[1635], block_u_data[1636], block_u_data[1637], block_u_data[1638], block_u_data[1639], block_u_data[1640], block_u_data[1641], block_u_data[1642], block_u_data[1643], block_u_data[1644], block_u_data[1645], block_u_data[1646], block_u_data[1647], block_u_data[1648], block_u_data[1649], block_u_data[1650], block_u_data[1651], block_u_data[1652], block_u_data[1653], block_u_data[1654], block_u_data[1655], block_u_data[1656], block_u_data[1657], block_u_data[1658], block_u_data[1659], block_u_data[1660], block_u_data[1661], block_u_data[1662], block_u_data[1663], block_u_data[1664], block_u_data[1665], block_u_data[1666], block_u_data[1667], block_u_data[1668], block_u_data[1669], block_u_data[1670], block_u_data[1671], block_u_data[1672], block_u_data[1673], block_u_data[1674], block_u_data[1675], block_u_data[1676], block_u_data[1677], block_u_data[1678], block_u_data[1679], block_u_data[1680], block_u_data[1681], block_u_data[1682], block_u_data[1683], block_u_data[1684], block_u_data[1685], block_u_data[1686], block_u_data[1687], block_u_data[1688], block_u_data[1689], block_u_data[1690], block_u_data[1691], block_u_data[1692], block_u_data[1693], block_u_data[1694], block_u_data[1695], block_u_data[1696], block_u_data[1697], block_u_data[1698], block_u_data[1699], block_u_data[1700], block_u_data[1701], block_u_data[1702], block_u_data[1703], block_u_data[1704], block_u_data[1705], block_u_data[1706], block_u_data[1707], block_u_data[1708], block_u_data[1709], block_u_data[1710], block_u_data[1711], block_u_data[1712], block_u_data[1713], block_u_data[1714], block_u_data[1715], block_u_data[1716], block_u_data[1717], block_u_data[1718], block_u_data[1719], block_u_data[1720], block_u_data[1721], block_u_data[1722], block_u_data[1723], block_u_data[1724], block_u_data[1725], block_u_data[1726], block_u_data[1727], block_u_data[1728], block_u_data[1729], block_u_data[1730], block_u_data[1731], block_u_data[1732], block_u_data[1733], block_u_data[1734], block_u_data[1735], block_u_data[1736], block_u_data[1737], block_u_data[1738], block_u_data[1739], block_u_data[1740], block_u_data[1741], block_u_data[1742], block_u_data[1743], block_u_data[1744], block_u_data[1745], block_u_data[1746], block_u_data[1747], block_u_data[1748], block_u_data[1749], block_u_data[1750], block_u_data[1751], block_u_data[1752], block_u_data[1753], block_u_data[1754], block_u_data[1755], block_u_data[1756], block_u_data[1757], block_u_data[1758], block_u_data[1759], block_u_data[1760], block_u_data[1761], block_u_data[1762], block_u_data[1763], block_u_data[1764], block_u_data[1765], block_u_data[1766], block_u_data[1767], block_u_data[1768], block_u_data[1769], block_u_data[1770], block_u_data[1771], block_u_data[1772], block_u_data[1773], block_u_data[1774], block_u_data[1775], block_u_data[1776], block_u_data[1777], block_u_data[1778], block_u_data[1779], block_u_data[1780], block_u_data[1781], block_u_data[1782], block_u_data[1783], block_u_data[1784], block_u_data[1785], block_u_data[1786], block_u_data[1787], block_u_data[1788], block_u_data[1789], block_u_data[1790], block_u_data[1791], block_u_data[1792], block_u_data[1793], block_u_data[1794], block_u_data[1795], block_u_data[1796], block_u_data[1797], block_u_data[1798], block_u_data[1799], block_u_data[1800], block_u_data[1801], block_u_data[1802], block_u_data[1803], block_u_data[1804], block_u_data[1805], block_u_data[1806], block_u_data[1807], block_u_data[1808], block_u_data[1809], block_u_data[1810], block_u_data[1811], block_u_data[1812], block_u_data[1813], block_u_data[1814], block_u_data[1815], block_u_data[1816], block_u_data[1817], block_u_data[1818], block_u_data[1819], block_u_data[1820], block_u_data[1821], block_u_data[1822], block_u_data[1823], block_u_data[1824], block_u_data[1825], block_u_data[1826], block_u_data[1827], block_u_data[1828], block_u_data[1829], block_u_data[1830], block_u_data[1831], block_u_data[1832], block_u_data[1833], block_u_data[1834], block_u_data[1835], block_u_data[1836], block_u_data[1837], block_u_data[1838], block_u_data[1839], block_u_data[1840], block_u_data[1841], block_u_data[1842], block_u_data[1843], block_u_data[1844], block_u_data[1845], block_u_data[1846], block_u_data[1847], block_u_data[1848], block_u_data[1849], block_u_data[1850], block_u_data[1851], block_u_data[1852], block_u_data[1853], block_u_data[1854], block_u_data[1855], block_u_data[1856], block_u_data[1857], block_u_data[1858], block_u_data[1859], block_u_data[1860], block_u_data[1861], block_u_data[1862], block_u_data[1863], block_u_data[1864], block_u_data[1865], block_u_data[1866], block_u_data[1867], block_u_data[1868], block_u_data[1869], block_u_data[1870], block_u_data[1871], block_u_data[1872], block_u_data[1873], block_u_data[1874], block_u_data[1875], block_u_data[1876], block_u_data[1877], block_u_data[1878], block_u_data[1879], block_u_data[1880], block_u_data[1881], block_u_data[1882], block_u_data[1883], block_u_data[1884], block_u_data[1885], block_u_data[1886], block_u_data[1887], block_u_data[1888], block_u_data[1889], block_u_data[1890], block_u_data[1891], block_u_data[1892], block_u_data[1893], block_u_data[1894], block_u_data[1895], block_u_data[1896], block_u_data[1897], block_u_data[1898], block_u_data[1899], block_u_data[1900], block_u_data[1901], block_u_data[1902], block_u_data[1903], block_u_data[1904], block_u_data[1905], block_u_data[1906], block_u_data[1907], block_u_data[1908], block_u_data[1909], block_u_data[1910], block_u_data[1911], block_u_data[1912], block_u_data[1913], block_u_data[1914], block_u_data[1915], block_u_data[1916], block_u_data[1917], block_u_data[1918], block_u_data[1919], block_u_data[1920], block_u_data[1921], block_u_data[1922], block_u_data[1923], block_u_data[1924], block_u_data[1925], block_u_data[1926], block_u_data[1927], block_u_data[1928], block_u_data[1929], block_u_data[1930], block_u_data[1931], block_u_data[1932], block_u_data[1933], block_u_data[1934], block_u_data[1935], block_u_data[1936], block_u_data[1937], block_u_data[1938], block_u_data[1939], block_u_data[1940], block_u_data[1941], block_u_data[1942], block_u_data[1943], block_u_data[1944], block_u_data[1945], block_u_data[1946], block_u_data[1947], block_u_data[1948], block_u_data[1949], block_u_data[1950], block_u_data[1951], block_u_data[1952], block_u_data[1953], block_u_data[1954], block_u_data[1955], block_u_data[1956], block_u_data[1957], block_u_data[1958], block_u_data[1959], block_u_data[1960], block_u_data[1961], block_u_data[1962], block_u_data[1963], block_u_data[1964], block_u_data[1965], block_u_data[1966], block_u_data[1967], block_u_data[1968], block_u_data[1969], block_u_data[1970], block_u_data[1971], block_u_data[1972], block_u_data[1973], block_u_data[1974], block_u_data[1975], block_u_data[1976], block_u_data[1977], block_u_data[1978], block_u_data[1979], block_u_data[1980], block_u_data[1981], block_u_data[1982], block_u_data[1983], block_u_data[1984], block_u_data[1985], block_u_data[1986], block_u_data[1987], block_u_data[1988], block_u_data[1989], block_u_data[1990], block_u_data[1991], block_u_data[1992], block_u_data[1993], block_u_data[1994], block_u_data[1995], block_u_data[1996], block_u_data[1997], block_u_data[1998], block_u_data[1999], block_u_data[2000], block_u_data[2001], block_u_data[2002], block_u_data[2003], block_u_data[2004], block_u_data[2005], block_u_data[2006], block_u_data[2007], block_u_data[2008], block_u_data[2009], block_u_data[2010], block_u_data[2011], block_u_data[2012], block_u_data[2013], block_u_data[2014], block_u_data[2015], block_u_data[2016], block_u_data[2017], block_u_data[2018], block_u_data[2019], block_u_data[2020], block_u_data[2021], block_u_data[2022], block_u_data[2023], block_u_data[2024], block_u_data[2025], block_u_data[2026], block_u_data[2027], block_u_data[2028], block_u_data[2029], block_u_data[2030], block_u_data[2031], block_u_data[2032], block_u_data[2033], block_u_data[2034], block_u_data[2035], block_u_data[2036], block_u_data[2037], block_u_data[2038], block_u_data[2039], block_u_data[2040], block_u_data[2041], block_u_data[2042], block_u_data[2043], block_u_data[2044], block_u_data[2045], block_u_data[2046], block_u_data[2047], block_u_data[2048], block_u_data[2049], block_u_data[2050], block_u_data[2051], block_u_data[2052], block_u_data[2053], block_u_data[2054], block_u_data[2055], block_u_data[2056], block_u_data[2057], block_u_data[2058], block_u_data[2059], block_u_data[2060], block_u_data[2061], block_u_data[2062], block_u_data[2063], block_u_data[2064], block_u_data[2065], block_u_data[2066], block_u_data[2067], block_u_data[2068], block_u_data[2069], block_u_data[2070], block_u_data[2071], block_u_data[2072], block_u_data[2073], block_u_data[2074], block_u_data[2075], block_u_data[2076], block_u_data[2077], block_u_data[2078], block_u_data[2079], block_u_data[2080], block_u_data[2081], block_u_data[2082], block_u_data[2083], block_u_data[2084], block_u_data[2085], block_u_data[2086], block_u_data[2087], block_u_data[2088], block_u_data[2089], block_u_data[2090], block_u_data[2091], block_u_data[2092], block_u_data[2093], block_u_data[2094], block_u_data[2095], block_u_data[2096], block_u_data[2097], block_u_data[2098], block_u_data[2099], block_u_data[2100], block_u_data[2101], block_u_data[2102], block_u_data[2103], block_u_data[2104], block_u_data[2105], block_u_data[2106], block_u_data[2107], block_u_data[2108], block_u_data[2109], block_u_data[2110], block_u_data[2111], block_u_data[2112], block_u_data[2113], block_u_data[2114], block_u_data[2115], block_u_data[2116], block_u_data[2117], block_u_data[2118], block_u_data[2119], block_u_data[2120], block_u_data[2121], block_u_data[2122], block_u_data[2123], block_u_data[2124], block_u_data[2125], block_u_data[2126], block_u_data[2127], block_u_data[2128], block_u_data[2129], block_u_data[2130], block_u_data[2131], block_u_data[2132], block_u_data[2133], block_u_data[2134], block_u_data[2135], block_u_data[2136], block_u_data[2137], block_u_data[2138], block_u_data[2139], block_u_data[2140], block_u_data[2141], block_u_data[2142], block_u_data[2143], block_u_data[2144], block_u_data[2145], block_u_data[2146], block_u_data[2147], block_u_data[2148], block_u_data[2149], block_u_data[2150], block_u_data[2151], block_u_data[2152], block_u_data[2153], block_u_data[2154], block_u_data[2155], block_u_data[2156], block_u_data[2157], block_u_data[2158], block_u_data[2159], block_u_data[2160], block_u_data[2161], block_u_data[2162], block_u_data[2163], block_u_data[2164], block_u_data[2165], block_u_data[2166], block_u_data[2167], block_u_data[2168], block_u_data[2169], block_u_data[2170], block_u_data[2171], block_u_data[2172], block_u_data[2173], block_u_data[2174], block_u_data[2175], block_u_data[2176], block_u_data[2177], block_u_data[2178], block_u_data[2179], block_u_data[2180], block_u_data[2181], block_u_data[2182], block_u_data[2183], block_u_data[2184], block_u_data[2185], block_u_data[2186]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
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

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728], block_u_data[729], block_u_data[730], block_u_data[731], block_u_data[732], block_u_data[733], block_u_data[734], block_u_data[735], block_u_data[736], block_u_data[737], block_u_data[738], block_u_data[739], block_u_data[740], block_u_data[741], block_u_data[742], block_u_data[743], block_u_data[744], block_u_data[745], block_u_data[746], block_u_data[747], block_u_data[748], block_u_data[749], block_u_data[750], block_u_data[751], block_u_data[752], block_u_data[753], block_u_data[754], block_u_data[755], block_u_data[756], block_u_data[757], block_u_data[758], block_u_data[759], block_u_data[760], block_u_data[761], block_u_data[762], block_u_data[763], block_u_data[764], block_u_data[765], block_u_data[766], block_u_data[767], block_u_data[768], block_u_data[769], block_u_data[770], block_u_data[771], block_u_data[772], block_u_data[773], block_u_data[774], block_u_data[775], block_u_data[776], block_u_data[777], block_u_data[778], block_u_data[779], block_u_data[780], block_u_data[781], block_u_data[782], block_u_data[783], block_u_data[784], block_u_data[785], block_u_data[786], block_u_data[787], block_u_data[788], block_u_data[789], block_u_data[790], block_u_data[791], block_u_data[792], block_u_data[793], block_u_data[794], block_u_data[795], block_u_data[796], block_u_data[797], block_u_data[798], block_u_data[799], block_u_data[800], block_u_data[801], block_u_data[802], block_u_data[803], block_u_data[804], block_u_data[805], block_u_data[806], block_u_data[807], block_u_data[808], block_u_data[809], block_u_data[810], block_u_data[811], block_u_data[812], block_u_data[813], block_u_data[814], block_u_data[815], block_u_data[816], block_u_data[817], block_u_data[818], block_u_data[819], block_u_data[820], block_u_data[821], block_u_data[822], block_u_data[823], block_u_data[824], block_u_data[825], block_u_data[826], block_u_data[827], block_u_data[828], block_u_data[829], block_u_data[830], block_u_data[831], block_u_data[832], block_u_data[833], block_u_data[834], block_u_data[835], block_u_data[836], block_u_data[837], block_u_data[838], block_u_data[839], block_u_data[840], block_u_data[841], block_u_data[842], block_u_data[843], block_u_data[844], block_u_data[845], block_u_data[846], block_u_data[847], block_u_data[848], block_u_data[849], block_u_data[850], block_u_data[851], block_u_data[852], block_u_data[853], block_u_data[854], block_u_data[855], block_u_data[856], block_u_data[857], block_u_data[858], block_u_data[859], block_u_data[860], block_u_data[861], block_u_data[862], block_u_data[863], block_u_data[864], block_u_data[865], block_u_data[866], block_u_data[867], block_u_data[868], block_u_data[869], block_u_data[870], block_u_data[871], block_u_data[872], block_u_data[873], block_u_data[874], block_u_data[875], block_u_data[876], block_u_data[877], block_u_data[878], block_u_data[879], block_u_data[880], block_u_data[881], block_u_data[882], block_u_data[883], block_u_data[884], block_u_data[885], block_u_data[886], block_u_data[887], block_u_data[888], block_u_data[889], block_u_data[890], block_u_data[891], block_u_data[892], block_u_data[893], block_u_data[894], block_u_data[895], block_u_data[896], block_u_data[897], block_u_data[898], block_u_data[899], block_u_data[900], block_u_data[901], block_u_data[902], block_u_data[903], block_u_data[904], block_u_data[905], block_u_data[906], block_u_data[907], block_u_data[908], block_u_data[909], block_u_data[910], block_u_data[911], block_u_data[912], block_u_data[913], block_u_data[914], block_u_data[915], block_u_data[916], block_u_data[917], block_u_data[918], block_u_data[919], block_u_data[920], block_u_data[921], block_u_data[922], block_u_data[923], block_u_data[924], block_u_data[925], block_u_data[926], block_u_data[927], block_u_data[928], block_u_data[929], block_u_data[930], block_u_data[931], block_u_data[932], block_u_data[933], block_u_data[934], block_u_data[935], block_u_data[936], block_u_data[937], block_u_data[938], block_u_data[939], block_u_data[940], block_u_data[941], block_u_data[942], block_u_data[943], block_u_data[944], block_u_data[945], block_u_data[946], block_u_data[947], block_u_data[948], block_u_data[949], block_u_data[950], block_u_data[951], block_u_data[952], block_u_data[953], block_u_data[954], block_u_data[955], block_u_data[956], block_u_data[957], block_u_data[958], block_u_data[959], block_u_data[960], block_u_data[961], block_u_data[962], block_u_data[963], block_u_data[964], block_u_data[965], block_u_data[966], block_u_data[967], block_u_data[968], block_u_data[969], block_u_data[970], block_u_data[971], block_u_data[972], block_u_data[973], block_u_data[974], block_u_data[975], block_u_data[976], block_u_data[977], block_u_data[978], block_u_data[979], block_u_data[980], block_u_data[981], block_u_data[982], block_u_data[983], block_u_data[984], block_u_data[985], block_u_data[986], block_u_data[987], block_u_data[988], block_u_data[989], block_u_data[990], block_u_data[991], block_u_data[992], block_u_data[993], block_u_data[994], block_u_data[995], block_u_data[996], block_u_data[997], block_u_data[998], block_u_data[999], block_u_data[1000], block_u_data[1001], block_u_data[1002], block_u_data[1003], block_u_data[1004], block_u_data[1005], block_u_data[1006], block_u_data[1007], block_u_data[1008], block_u_data[1009], block_u_data[1010], block_u_data[1011], block_u_data[1012], block_u_data[1013], block_u_data[1014], block_u_data[1015], block_u_data[1016], block_u_data[1017], block_u_data[1018], block_u_data[1019], block_u_data[1020], block_u_data[1021], block_u_data[1022], block_u_data[1023], block_u_data[1024], block_u_data[1025], block_u_data[1026], block_u_data[1027], block_u_data[1028], block_u_data[1029], block_u_data[1030], block_u_data[1031], block_u_data[1032], block_u_data[1033], block_u_data[1034], block_u_data[1035], block_u_data[1036], block_u_data[1037], block_u_data[1038], block_u_data[1039], block_u_data[1040], block_u_data[1041], block_u_data[1042], block_u_data[1043], block_u_data[1044], block_u_data[1045], block_u_data[1046], block_u_data[1047], block_u_data[1048], block_u_data[1049], block_u_data[1050], block_u_data[1051], block_u_data[1052], block_u_data[1053], block_u_data[1054], block_u_data[1055], block_u_data[1056], block_u_data[1057], block_u_data[1058], block_u_data[1059], block_u_data[1060], block_u_data[1061], block_u_data[1062], block_u_data[1063], block_u_data[1064], block_u_data[1065], block_u_data[1066], block_u_data[1067], block_u_data[1068], block_u_data[1069], block_u_data[1070], block_u_data[1071], block_u_data[1072], block_u_data[1073], block_u_data[1074], block_u_data[1075], block_u_data[1076], block_u_data[1077], block_u_data[1078], block_u_data[1079], block_u_data[1080], block_u_data[1081], block_u_data[1082], block_u_data[1083], block_u_data[1084], block_u_data[1085], block_u_data[1086], block_u_data[1087], block_u_data[1088], block_u_data[1089], block_u_data[1090], block_u_data[1091], block_u_data[1092], block_u_data[1093], block_u_data[1094], block_u_data[1095], block_u_data[1096], block_u_data[1097], block_u_data[1098], block_u_data[1099], block_u_data[1100], block_u_data[1101], block_u_data[1102], block_u_data[1103], block_u_data[1104], block_u_data[1105], block_u_data[1106], block_u_data[1107], block_u_data[1108], block_u_data[1109], block_u_data[1110], block_u_data[1111], block_u_data[1112], block_u_data[1113], block_u_data[1114], block_u_data[1115], block_u_data[1116], block_u_data[1117], block_u_data[1118], block_u_data[1119], block_u_data[1120], block_u_data[1121], block_u_data[1122], block_u_data[1123], block_u_data[1124], block_u_data[1125], block_u_data[1126], block_u_data[1127], block_u_data[1128], block_u_data[1129], block_u_data[1130], block_u_data[1131], block_u_data[1132], block_u_data[1133], block_u_data[1134], block_u_data[1135], block_u_data[1136], block_u_data[1137], block_u_data[1138], block_u_data[1139], block_u_data[1140], block_u_data[1141], block_u_data[1142], block_u_data[1143], block_u_data[1144], block_u_data[1145], block_u_data[1146], block_u_data[1147], block_u_data[1148], block_u_data[1149], block_u_data[1150], block_u_data[1151], block_u_data[1152], block_u_data[1153], block_u_data[1154], block_u_data[1155], block_u_data[1156], block_u_data[1157], block_u_data[1158], block_u_data[1159], block_u_data[1160], block_u_data[1161], block_u_data[1162], block_u_data[1163], block_u_data[1164], block_u_data[1165], block_u_data[1166], block_u_data[1167], block_u_data[1168], block_u_data[1169], block_u_data[1170], block_u_data[1171], block_u_data[1172], block_u_data[1173], block_u_data[1174], block_u_data[1175], block_u_data[1176], block_u_data[1177], block_u_data[1178], block_u_data[1179], block_u_data[1180], block_u_data[1181], block_u_data[1182], block_u_data[1183], block_u_data[1184], block_u_data[1185], block_u_data[1186], block_u_data[1187], block_u_data[1188], block_u_data[1189], block_u_data[1190], block_u_data[1191], block_u_data[1192], block_u_data[1193], block_u_data[1194], block_u_data[1195], block_u_data[1196], block_u_data[1197], block_u_data[1198], block_u_data[1199], block_u_data[1200], block_u_data[1201], block_u_data[1202], block_u_data[1203], block_u_data[1204], block_u_data[1205], block_u_data[1206], block_u_data[1207], block_u_data[1208], block_u_data[1209], block_u_data[1210], block_u_data[1211], block_u_data[1212], block_u_data[1213], block_u_data[1214], block_u_data[1215], block_u_data[1216], block_u_data[1217], block_u_data[1218], block_u_data[1219], block_u_data[1220], block_u_data[1221], block_u_data[1222], block_u_data[1223], block_u_data[1224], block_u_data[1225], block_u_data[1226], block_u_data[1227], block_u_data[1228], block_u_data[1229], block_u_data[1230], block_u_data[1231], block_u_data[1232], block_u_data[1233], block_u_data[1234], block_u_data[1235], block_u_data[1236], block_u_data[1237], block_u_data[1238], block_u_data[1239], block_u_data[1240], block_u_data[1241], block_u_data[1242], block_u_data[1243], block_u_data[1244], block_u_data[1245], block_u_data[1246], block_u_data[1247], block_u_data[1248], block_u_data[1249], block_u_data[1250], block_u_data[1251], block_u_data[1252], block_u_data[1253], block_u_data[1254], block_u_data[1255], block_u_data[1256], block_u_data[1257], block_u_data[1258], block_u_data[1259], block_u_data[1260], block_u_data[1261], block_u_data[1262], block_u_data[1263], block_u_data[1264], block_u_data[1265], block_u_data[1266], block_u_data[1267], block_u_data[1268], block_u_data[1269], block_u_data[1270], block_u_data[1271], block_u_data[1272], block_u_data[1273], block_u_data[1274], block_u_data[1275], block_u_data[1276], block_u_data[1277], block_u_data[1278], block_u_data[1279], block_u_data[1280], block_u_data[1281], block_u_data[1282], block_u_data[1283], block_u_data[1284], block_u_data[1285], block_u_data[1286], block_u_data[1287], block_u_data[1288], block_u_data[1289], block_u_data[1290], block_u_data[1291], block_u_data[1292], block_u_data[1293], block_u_data[1294], block_u_data[1295], block_u_data[1296], block_u_data[1297], block_u_data[1298], block_u_data[1299], block_u_data[1300], block_u_data[1301], block_u_data[1302], block_u_data[1303], block_u_data[1304], block_u_data[1305], block_u_data[1306], block_u_data[1307], block_u_data[1308], block_u_data[1309], block_u_data[1310], block_u_data[1311], block_u_data[1312], block_u_data[1313], block_u_data[1314], block_u_data[1315], block_u_data[1316], block_u_data[1317], block_u_data[1318], block_u_data[1319], block_u_data[1320], block_u_data[1321], block_u_data[1322], block_u_data[1323], block_u_data[1324], block_u_data[1325], block_u_data[1326], block_u_data[1327], block_u_data[1328], block_u_data[1329], block_u_data[1330], block_u_data[1331], block_u_data[1332], block_u_data[1333], block_u_data[1334], block_u_data[1335], block_u_data[1336], block_u_data[1337], block_u_data[1338], block_u_data[1339], block_u_data[1340], block_u_data[1341], block_u_data[1342], block_u_data[1343], block_u_data[1344], block_u_data[1345], block_u_data[1346], block_u_data[1347], block_u_data[1348], block_u_data[1349], block_u_data[1350], block_u_data[1351], block_u_data[1352], block_u_data[1353], block_u_data[1354], block_u_data[1355], block_u_data[1356], block_u_data[1357], block_u_data[1358], block_u_data[1359], block_u_data[1360], block_u_data[1361], block_u_data[1362], block_u_data[1363], block_u_data[1364], block_u_data[1365], block_u_data[1366], block_u_data[1367], block_u_data[1368], block_u_data[1369], block_u_data[1370], block_u_data[1371], block_u_data[1372], block_u_data[1373], block_u_data[1374], block_u_data[1375], block_u_data[1376], block_u_data[1377], block_u_data[1378], block_u_data[1379], block_u_data[1380], block_u_data[1381], block_u_data[1382], block_u_data[1383], block_u_data[1384], block_u_data[1385], block_u_data[1386], block_u_data[1387], block_u_data[1388], block_u_data[1389], block_u_data[1390], block_u_data[1391], block_u_data[1392], block_u_data[1393], block_u_data[1394], block_u_data[1395], block_u_data[1396], block_u_data[1397], block_u_data[1398], block_u_data[1399], block_u_data[1400], block_u_data[1401], block_u_data[1402], block_u_data[1403], block_u_data[1404], block_u_data[1405], block_u_data[1406], block_u_data[1407], block_u_data[1408], block_u_data[1409], block_u_data[1410], block_u_data[1411], block_u_data[1412], block_u_data[1413], block_u_data[1414], block_u_data[1415], block_u_data[1416], block_u_data[1417], block_u_data[1418], block_u_data[1419], block_u_data[1420], block_u_data[1421], block_u_data[1422], block_u_data[1423], block_u_data[1424], block_u_data[1425], block_u_data[1426], block_u_data[1427], block_u_data[1428], block_u_data[1429], block_u_data[1430], block_u_data[1431], block_u_data[1432], block_u_data[1433], block_u_data[1434], block_u_data[1435], block_u_data[1436], block_u_data[1437], block_u_data[1438], block_u_data[1439], block_u_data[1440], block_u_data[1441], block_u_data[1442], block_u_data[1443], block_u_data[1444], block_u_data[1445], block_u_data[1446], block_u_data[1447], block_u_data[1448], block_u_data[1449], block_u_data[1450], block_u_data[1451], block_u_data[1452], block_u_data[1453], block_u_data[1454], block_u_data[1455], block_u_data[1456], block_u_data[1457], block_u_data[1458], block_u_data[1459], block_u_data[1460], block_u_data[1461], block_u_data[1462], block_u_data[1463], block_u_data[1464], block_u_data[1465], block_u_data[1466], block_u_data[1467], block_u_data[1468], block_u_data[1469], block_u_data[1470], block_u_data[1471], block_u_data[1472], block_u_data[1473], block_u_data[1474], block_u_data[1475], block_u_data[1476], block_u_data[1477], block_u_data[1478], block_u_data[1479], block_u_data[1480], block_u_data[1481], block_u_data[1482], block_u_data[1483], block_u_data[1484], block_u_data[1485], block_u_data[1486], block_u_data[1487], block_u_data[1488], block_u_data[1489], block_u_data[1490], block_u_data[1491], block_u_data[1492], block_u_data[1493], block_u_data[1494], block_u_data[1495], block_u_data[1496], block_u_data[1497], block_u_data[1498], block_u_data[1499], block_u_data[1500], block_u_data[1501], block_u_data[1502], block_u_data[1503], block_u_data[1504], block_u_data[1505], block_u_data[1506], block_u_data[1507], block_u_data[1508], block_u_data[1509], block_u_data[1510], block_u_data[1511], block_u_data[1512], block_u_data[1513], block_u_data[1514], block_u_data[1515], block_u_data[1516], block_u_data[1517], block_u_data[1518], block_u_data[1519], block_u_data[1520], block_u_data[1521], block_u_data[1522], block_u_data[1523], block_u_data[1524], block_u_data[1525], block_u_data[1526], block_u_data[1527], block_u_data[1528], block_u_data[1529], block_u_data[1530], block_u_data[1531], block_u_data[1532], block_u_data[1533], block_u_data[1534], block_u_data[1535], block_u_data[1536], block_u_data[1537], block_u_data[1538], block_u_data[1539], block_u_data[1540], block_u_data[1541], block_u_data[1542], block_u_data[1543], block_u_data[1544], block_u_data[1545], block_u_data[1546], block_u_data[1547], block_u_data[1548], block_u_data[1549], block_u_data[1550], block_u_data[1551], block_u_data[1552], block_u_data[1553], block_u_data[1554], block_u_data[1555], block_u_data[1556], block_u_data[1557], block_u_data[1558], block_u_data[1559], block_u_data[1560], block_u_data[1561], block_u_data[1562], block_u_data[1563], block_u_data[1564], block_u_data[1565], block_u_data[1566], block_u_data[1567], block_u_data[1568], block_u_data[1569], block_u_data[1570], block_u_data[1571], block_u_data[1572], block_u_data[1573], block_u_data[1574], block_u_data[1575], block_u_data[1576], block_u_data[1577], block_u_data[1578], block_u_data[1579], block_u_data[1580], block_u_data[1581], block_u_data[1582], block_u_data[1583], block_u_data[1584], block_u_data[1585], block_u_data[1586], block_u_data[1587], block_u_data[1588], block_u_data[1589], block_u_data[1590], block_u_data[1591], block_u_data[1592], block_u_data[1593], block_u_data[1594], block_u_data[1595], block_u_data[1596], block_u_data[1597], block_u_data[1598], block_u_data[1599], block_u_data[1600], block_u_data[1601], block_u_data[1602], block_u_data[1603], block_u_data[1604], block_u_data[1605], block_u_data[1606], block_u_data[1607], block_u_data[1608], block_u_data[1609], block_u_data[1610], block_u_data[1611], block_u_data[1612], block_u_data[1613], block_u_data[1614], block_u_data[1615], block_u_data[1616], block_u_data[1617], block_u_data[1618], block_u_data[1619], block_u_data[1620], block_u_data[1621], block_u_data[1622], block_u_data[1623], block_u_data[1624], block_u_data[1625], block_u_data[1626], block_u_data[1627], block_u_data[1628], block_u_data[1629], block_u_data[1630], block_u_data[1631], block_u_data[1632], block_u_data[1633], block_u_data[1634], block_u_data[1635], block_u_data[1636], block_u_data[1637], block_u_data[1638], block_u_data[1639], block_u_data[1640], block_u_data[1641], block_u_data[1642], block_u_data[1643], block_u_data[1644], block_u_data[1645], block_u_data[1646], block_u_data[1647], block_u_data[1648], block_u_data[1649], block_u_data[1650], block_u_data[1651], block_u_data[1652], block_u_data[1653], block_u_data[1654], block_u_data[1655], block_u_data[1656], block_u_data[1657], block_u_data[1658], block_u_data[1659], block_u_data[1660], block_u_data[1661], block_u_data[1662], block_u_data[1663], block_u_data[1664], block_u_data[1665], block_u_data[1666], block_u_data[1667], block_u_data[1668], block_u_data[1669], block_u_data[1670], block_u_data[1671], block_u_data[1672], block_u_data[1673], block_u_data[1674], block_u_data[1675], block_u_data[1676], block_u_data[1677], block_u_data[1678], block_u_data[1679], block_u_data[1680], block_u_data[1681], block_u_data[1682], block_u_data[1683], block_u_data[1684], block_u_data[1685], block_u_data[1686], block_u_data[1687], block_u_data[1688], block_u_data[1689], block_u_data[1690], block_u_data[1691], block_u_data[1692], block_u_data[1693], block_u_data[1694], block_u_data[1695], block_u_data[1696], block_u_data[1697], block_u_data[1698], block_u_data[1699], block_u_data[1700], block_u_data[1701], block_u_data[1702], block_u_data[1703], block_u_data[1704], block_u_data[1705], block_u_data[1706], block_u_data[1707], block_u_data[1708], block_u_data[1709], block_u_data[1710], block_u_data[1711], block_u_data[1712], block_u_data[1713], block_u_data[1714], block_u_data[1715], block_u_data[1716], block_u_data[1717], block_u_data[1718], block_u_data[1719], block_u_data[1720], block_u_data[1721], block_u_data[1722], block_u_data[1723], block_u_data[1724], block_u_data[1725], block_u_data[1726], block_u_data[1727], block_u_data[1728], block_u_data[1729], block_u_data[1730], block_u_data[1731], block_u_data[1732], block_u_data[1733], block_u_data[1734], block_u_data[1735], block_u_data[1736], block_u_data[1737], block_u_data[1738], block_u_data[1739], block_u_data[1740], block_u_data[1741], block_u_data[1742], block_u_data[1743], block_u_data[1744], block_u_data[1745], block_u_data[1746], block_u_data[1747], block_u_data[1748], block_u_data[1749], block_u_data[1750], block_u_data[1751], block_u_data[1752], block_u_data[1753], block_u_data[1754], block_u_data[1755], block_u_data[1756], block_u_data[1757], block_u_data[1758], block_u_data[1759], block_u_data[1760], block_u_data[1761], block_u_data[1762], block_u_data[1763], block_u_data[1764], block_u_data[1765], block_u_data[1766], block_u_data[1767], block_u_data[1768], block_u_data[1769], block_u_data[1770], block_u_data[1771], block_u_data[1772], block_u_data[1773], block_u_data[1774], block_u_data[1775], block_u_data[1776], block_u_data[1777], block_u_data[1778], block_u_data[1779], block_u_data[1780], block_u_data[1781], block_u_data[1782], block_u_data[1783], block_u_data[1784], block_u_data[1785], block_u_data[1786], block_u_data[1787], block_u_data[1788], block_u_data[1789], block_u_data[1790], block_u_data[1791], block_u_data[1792], block_u_data[1793], block_u_data[1794], block_u_data[1795], block_u_data[1796], block_u_data[1797], block_u_data[1798], block_u_data[1799], block_u_data[1800], block_u_data[1801], block_u_data[1802], block_u_data[1803], block_u_data[1804], block_u_data[1805], block_u_data[1806], block_u_data[1807], block_u_data[1808], block_u_data[1809], block_u_data[1810], block_u_data[1811], block_u_data[1812], block_u_data[1813], block_u_data[1814], block_u_data[1815], block_u_data[1816], block_u_data[1817], block_u_data[1818], block_u_data[1819], block_u_data[1820], block_u_data[1821], block_u_data[1822], block_u_data[1823], block_u_data[1824], block_u_data[1825], block_u_data[1826], block_u_data[1827], block_u_data[1828], block_u_data[1829], block_u_data[1830], block_u_data[1831], block_u_data[1832], block_u_data[1833], block_u_data[1834], block_u_data[1835], block_u_data[1836], block_u_data[1837], block_u_data[1838], block_u_data[1839], block_u_data[1840], block_u_data[1841], block_u_data[1842], block_u_data[1843], block_u_data[1844], block_u_data[1845], block_u_data[1846], block_u_data[1847], block_u_data[1848], block_u_data[1849], block_u_data[1850], block_u_data[1851], block_u_data[1852], block_u_data[1853], block_u_data[1854], block_u_data[1855], block_u_data[1856], block_u_data[1857], block_u_data[1858], block_u_data[1859], block_u_data[1860], block_u_data[1861], block_u_data[1862], block_u_data[1863], block_u_data[1864], block_u_data[1865], block_u_data[1866], block_u_data[1867], block_u_data[1868], block_u_data[1869], block_u_data[1870], block_u_data[1871], block_u_data[1872], block_u_data[1873], block_u_data[1874], block_u_data[1875], block_u_data[1876], block_u_data[1877], block_u_data[1878], block_u_data[1879], block_u_data[1880], block_u_data[1881], block_u_data[1882], block_u_data[1883], block_u_data[1884], block_u_data[1885], block_u_data[1886], block_u_data[1887], block_u_data[1888], block_u_data[1889], block_u_data[1890], block_u_data[1891], block_u_data[1892], block_u_data[1893], block_u_data[1894], block_u_data[1895], block_u_data[1896], block_u_data[1897], block_u_data[1898], block_u_data[1899], block_u_data[1900], block_u_data[1901], block_u_data[1902], block_u_data[1903], block_u_data[1904], block_u_data[1905], block_u_data[1906], block_u_data[1907], block_u_data[1908], block_u_data[1909], block_u_data[1910], block_u_data[1911], block_u_data[1912], block_u_data[1913], block_u_data[1914], block_u_data[1915], block_u_data[1916], block_u_data[1917], block_u_data[1918], block_u_data[1919], block_u_data[1920], block_u_data[1921], block_u_data[1922], block_u_data[1923], block_u_data[1924], block_u_data[1925], block_u_data[1926], block_u_data[1927], block_u_data[1928], block_u_data[1929], block_u_data[1930], block_u_data[1931], block_u_data[1932], block_u_data[1933], block_u_data[1934], block_u_data[1935], block_u_data[1936], block_u_data[1937], block_u_data[1938], block_u_data[1939], block_u_data[1940], block_u_data[1941], block_u_data[1942], block_u_data[1943], block_u_data[1944], block_u_data[1945], block_u_data[1946], block_u_data[1947], block_u_data[1948], block_u_data[1949], block_u_data[1950], block_u_data[1951], block_u_data[1952], block_u_data[1953], block_u_data[1954], block_u_data[1955], block_u_data[1956], block_u_data[1957], block_u_data[1958], block_u_data[1959], block_u_data[1960], block_u_data[1961], block_u_data[1962], block_u_data[1963], block_u_data[1964], block_u_data[1965], block_u_data[1966], block_u_data[1967], block_u_data[1968], block_u_data[1969], block_u_data[1970], block_u_data[1971], block_u_data[1972], block_u_data[1973], block_u_data[1974], block_u_data[1975], block_u_data[1976], block_u_data[1977], block_u_data[1978], block_u_data[1979], block_u_data[1980], block_u_data[1981], block_u_data[1982], block_u_data[1983], block_u_data[1984], block_u_data[1985], block_u_data[1986], block_u_data[1987], block_u_data[1988], block_u_data[1989], block_u_data[1990], block_u_data[1991], block_u_data[1992], block_u_data[1993], block_u_data[1994], block_u_data[1995], block_u_data[1996], block_u_data[1997], block_u_data[1998], block_u_data[1999], block_u_data[2000], block_u_data[2001], block_u_data[2002], block_u_data[2003], block_u_data[2004], block_u_data[2005], block_u_data[2006], block_u_data[2007], block_u_data[2008], block_u_data[2009], block_u_data[2010], block_u_data[2011], block_u_data[2012], block_u_data[2013], block_u_data[2014], block_u_data[2015], block_u_data[2016], block_u_data[2017], block_u_data[2018], block_u_data[2019], block_u_data[2020], block_u_data[2021], block_u_data[2022], block_u_data[2023], block_u_data[2024], block_u_data[2025], block_u_data[2026], block_u_data[2027], block_u_data[2028], block_u_data[2029], block_u_data[2030], block_u_data[2031], block_u_data[2032], block_u_data[2033], block_u_data[2034], block_u_data[2035], block_u_data[2036], block_u_data[2037], block_u_data[2038], block_u_data[2039], block_u_data[2040], block_u_data[2041], block_u_data[2042], block_u_data[2043], block_u_data[2044], block_u_data[2045], block_u_data[2046], block_u_data[2047], block_u_data[2048], block_u_data[2049], block_u_data[2050], block_u_data[2051], block_u_data[2052], block_u_data[2053], block_u_data[2054], block_u_data[2055], block_u_data[2056], block_u_data[2057], block_u_data[2058], block_u_data[2059], block_u_data[2060], block_u_data[2061], block_u_data[2062], block_u_data[2063], block_u_data[2064], block_u_data[2065], block_u_data[2066], block_u_data[2067], block_u_data[2068], block_u_data[2069], block_u_data[2070], block_u_data[2071], block_u_data[2072], block_u_data[2073], block_u_data[2074], block_u_data[2075], block_u_data[2076], block_u_data[2077], block_u_data[2078], block_u_data[2079], block_u_data[2080], block_u_data[2081], block_u_data[2082], block_u_data[2083], block_u_data[2084], block_u_data[2085], block_u_data[2086], block_u_data[2087], block_u_data[2088], block_u_data[2089], block_u_data[2090], block_u_data[2091], block_u_data[2092], block_u_data[2093], block_u_data[2094], block_u_data[2095], block_u_data[2096], block_u_data[2097], block_u_data[2098], block_u_data[2099], block_u_data[2100], block_u_data[2101], block_u_data[2102], block_u_data[2103], block_u_data[2104], block_u_data[2105], block_u_data[2106], block_u_data[2107], block_u_data[2108], block_u_data[2109], block_u_data[2110], block_u_data[2111], block_u_data[2112], block_u_data[2113], block_u_data[2114], block_u_data[2115], block_u_data[2116], block_u_data[2117], block_u_data[2118], block_u_data[2119], block_u_data[2120], block_u_data[2121], block_u_data[2122], block_u_data[2123], block_u_data[2124], block_u_data[2125], block_u_data[2126], block_u_data[2127], block_u_data[2128], block_u_data[2129], block_u_data[2130], block_u_data[2131], block_u_data[2132], block_u_data[2133], block_u_data[2134], block_u_data[2135], block_u_data[2136], block_u_data[2137], block_u_data[2138], block_u_data[2139], block_u_data[2140], block_u_data[2141], block_u_data[2142], block_u_data[2143], block_u_data[2144], block_u_data[2145], block_u_data[2146], block_u_data[2147], block_u_data[2148], block_u_data[2149], block_u_data[2150], block_u_data[2151], block_u_data[2152], block_u_data[2153], block_u_data[2154], block_u_data[2155], block_u_data[2156], block_u_data[2157], block_u_data[2158], block_u_data[2159], block_u_data[2160], block_u_data[2161], block_u_data[2162], block_u_data[2163], block_u_data[2164], block_u_data[2165], block_u_data[2166], block_u_data[2167], block_u_data[2168], block_u_data[2169], block_u_data[2170], block_u_data[2171], block_u_data[2172], block_u_data[2173], block_u_data[2174], block_u_data[2175], block_u_data[2176], block_u_data[2177], block_u_data[2178], block_u_data[2179], block_u_data[2180], block_u_data[2181], block_u_data[2182], block_u_data[2183], block_u_data[2184], block_u_data[2185], block_u_data[2186]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728], block_u_data[729], block_u_data[730], block_u_data[731], block_u_data[732], block_u_data[733], block_u_data[734], block_u_data[735], block_u_data[736], block_u_data[737], block_u_data[738], block_u_data[739], block_u_data[740], block_u_data[741], block_u_data[742], block_u_data[743], block_u_data[744], block_u_data[745], block_u_data[746], block_u_data[747], block_u_data[748], block_u_data[749], block_u_data[750], block_u_data[751], block_u_data[752], block_u_data[753], block_u_data[754], block_u_data[755], block_u_data[756], block_u_data[757], block_u_data[758], block_u_data[759], block_u_data[760], block_u_data[761], block_u_data[762], block_u_data[763], block_u_data[764], block_u_data[765], block_u_data[766], block_u_data[767], block_u_data[768], block_u_data[769], block_u_data[770], block_u_data[771], block_u_data[772], block_u_data[773], block_u_data[774], block_u_data[775], block_u_data[776], block_u_data[777], block_u_data[778], block_u_data[779], block_u_data[780], block_u_data[781], block_u_data[782], block_u_data[783], block_u_data[784], block_u_data[785], block_u_data[786], block_u_data[787], block_u_data[788], block_u_data[789], block_u_data[790], block_u_data[791], block_u_data[792], block_u_data[793], block_u_data[794], block_u_data[795], block_u_data[796], block_u_data[797], block_u_data[798], block_u_data[799], block_u_data[800], block_u_data[801], block_u_data[802], block_u_data[803], block_u_data[804], block_u_data[805], block_u_data[806], block_u_data[807], block_u_data[808], block_u_data[809], block_u_data[810], block_u_data[811], block_u_data[812], block_u_data[813], block_u_data[814], block_u_data[815], block_u_data[816], block_u_data[817], block_u_data[818], block_u_data[819], block_u_data[820], block_u_data[821], block_u_data[822], block_u_data[823], block_u_data[824], block_u_data[825], block_u_data[826], block_u_data[827], block_u_data[828], block_u_data[829], block_u_data[830], block_u_data[831], block_u_data[832], block_u_data[833], block_u_data[834], block_u_data[835], block_u_data[836], block_u_data[837], block_u_data[838], block_u_data[839], block_u_data[840], block_u_data[841], block_u_data[842], block_u_data[843], block_u_data[844], block_u_data[845], block_u_data[846], block_u_data[847], block_u_data[848], block_u_data[849], block_u_data[850], block_u_data[851], block_u_data[852], block_u_data[853], block_u_data[854], block_u_data[855], block_u_data[856], block_u_data[857], block_u_data[858], block_u_data[859], block_u_data[860], block_u_data[861], block_u_data[862], block_u_data[863], block_u_data[864], block_u_data[865], block_u_data[866], block_u_data[867], block_u_data[868], block_u_data[869], block_u_data[870], block_u_data[871], block_u_data[872], block_u_data[873], block_u_data[874], block_u_data[875], block_u_data[876], block_u_data[877], block_u_data[878], block_u_data[879], block_u_data[880], block_u_data[881], block_u_data[882], block_u_data[883], block_u_data[884], block_u_data[885], block_u_data[886], block_u_data[887], block_u_data[888], block_u_data[889], block_u_data[890], block_u_data[891], block_u_data[892], block_u_data[893], block_u_data[894], block_u_data[895], block_u_data[896], block_u_data[897], block_u_data[898], block_u_data[899], block_u_data[900], block_u_data[901], block_u_data[902], block_u_data[903], block_u_data[904], block_u_data[905], block_u_data[906], block_u_data[907], block_u_data[908], block_u_data[909], block_u_data[910], block_u_data[911], block_u_data[912], block_u_data[913], block_u_data[914], block_u_data[915], block_u_data[916], block_u_data[917], block_u_data[918], block_u_data[919], block_u_data[920], block_u_data[921], block_u_data[922], block_u_data[923], block_u_data[924], block_u_data[925], block_u_data[926], block_u_data[927], block_u_data[928], block_u_data[929], block_u_data[930], block_u_data[931], block_u_data[932], block_u_data[933], block_u_data[934], block_u_data[935], block_u_data[936], block_u_data[937], block_u_data[938], block_u_data[939], block_u_data[940], block_u_data[941], block_u_data[942], block_u_data[943], block_u_data[944], block_u_data[945], block_u_data[946], block_u_data[947], block_u_data[948], block_u_data[949], block_u_data[950], block_u_data[951], block_u_data[952], block_u_data[953], block_u_data[954], block_u_data[955], block_u_data[956], block_u_data[957], block_u_data[958], block_u_data[959], block_u_data[960], block_u_data[961], block_u_data[962], block_u_data[963], block_u_data[964], block_u_data[965], block_u_data[966], block_u_data[967], block_u_data[968], block_u_data[969], block_u_data[970], block_u_data[971], block_u_data[972], block_u_data[973], block_u_data[974], block_u_data[975], block_u_data[976], block_u_data[977], block_u_data[978], block_u_data[979], block_u_data[980], block_u_data[981], block_u_data[982], block_u_data[983], block_u_data[984], block_u_data[985], block_u_data[986], block_u_data[987], block_u_data[988], block_u_data[989], block_u_data[990], block_u_data[991], block_u_data[992], block_u_data[993], block_u_data[994], block_u_data[995], block_u_data[996], block_u_data[997], block_u_data[998], block_u_data[999], block_u_data[1000], block_u_data[1001], block_u_data[1002], block_u_data[1003], block_u_data[1004], block_u_data[1005], block_u_data[1006], block_u_data[1007], block_u_data[1008], block_u_data[1009], block_u_data[1010], block_u_data[1011], block_u_data[1012], block_u_data[1013], block_u_data[1014], block_u_data[1015], block_u_data[1016], block_u_data[1017], block_u_data[1018], block_u_data[1019], block_u_data[1020], block_u_data[1021], block_u_data[1022], block_u_data[1023], block_u_data[1024], block_u_data[1025], block_u_data[1026], block_u_data[1027], block_u_data[1028], block_u_data[1029], block_u_data[1030], block_u_data[1031], block_u_data[1032], block_u_data[1033], block_u_data[1034], block_u_data[1035], block_u_data[1036], block_u_data[1037], block_u_data[1038], block_u_data[1039], block_u_data[1040], block_u_data[1041], block_u_data[1042], block_u_data[1043], block_u_data[1044], block_u_data[1045], block_u_data[1046], block_u_data[1047], block_u_data[1048], block_u_data[1049], block_u_data[1050], block_u_data[1051], block_u_data[1052], block_u_data[1053], block_u_data[1054], block_u_data[1055], block_u_data[1056], block_u_data[1057], block_u_data[1058], block_u_data[1059], block_u_data[1060], block_u_data[1061], block_u_data[1062], block_u_data[1063], block_u_data[1064], block_u_data[1065], block_u_data[1066], block_u_data[1067], block_u_data[1068], block_u_data[1069], block_u_data[1070], block_u_data[1071], block_u_data[1072], block_u_data[1073], block_u_data[1074], block_u_data[1075], block_u_data[1076], block_u_data[1077], block_u_data[1078], block_u_data[1079], block_u_data[1080], block_u_data[1081], block_u_data[1082], block_u_data[1083], block_u_data[1084], block_u_data[1085], block_u_data[1086], block_u_data[1087], block_u_data[1088], block_u_data[1089], block_u_data[1090], block_u_data[1091], block_u_data[1092], block_u_data[1093], block_u_data[1094], block_u_data[1095], block_u_data[1096], block_u_data[1097], block_u_data[1098], block_u_data[1099], block_u_data[1100], block_u_data[1101], block_u_data[1102], block_u_data[1103], block_u_data[1104], block_u_data[1105], block_u_data[1106], block_u_data[1107], block_u_data[1108], block_u_data[1109], block_u_data[1110], block_u_data[1111], block_u_data[1112], block_u_data[1113], block_u_data[1114], block_u_data[1115], block_u_data[1116], block_u_data[1117], block_u_data[1118], block_u_data[1119], block_u_data[1120], block_u_data[1121], block_u_data[1122], block_u_data[1123], block_u_data[1124], block_u_data[1125], block_u_data[1126], block_u_data[1127], block_u_data[1128], block_u_data[1129], block_u_data[1130], block_u_data[1131], block_u_data[1132], block_u_data[1133], block_u_data[1134], block_u_data[1135], block_u_data[1136], block_u_data[1137], block_u_data[1138], block_u_data[1139], block_u_data[1140], block_u_data[1141], block_u_data[1142], block_u_data[1143], block_u_data[1144], block_u_data[1145], block_u_data[1146], block_u_data[1147], block_u_data[1148], block_u_data[1149], block_u_data[1150], block_u_data[1151], block_u_data[1152], block_u_data[1153], block_u_data[1154], block_u_data[1155], block_u_data[1156], block_u_data[1157], block_u_data[1158], block_u_data[1159], block_u_data[1160], block_u_data[1161], block_u_data[1162], block_u_data[1163], block_u_data[1164], block_u_data[1165], block_u_data[1166], block_u_data[1167], block_u_data[1168], block_u_data[1169], block_u_data[1170], block_u_data[1171], block_u_data[1172], block_u_data[1173], block_u_data[1174], block_u_data[1175], block_u_data[1176], block_u_data[1177], block_u_data[1178], block_u_data[1179], block_u_data[1180], block_u_data[1181], block_u_data[1182], block_u_data[1183], block_u_data[1184], block_u_data[1185], block_u_data[1186], block_u_data[1187], block_u_data[1188], block_u_data[1189], block_u_data[1190], block_u_data[1191], block_u_data[1192], block_u_data[1193], block_u_data[1194], block_u_data[1195], block_u_data[1196], block_u_data[1197], block_u_data[1198], block_u_data[1199], block_u_data[1200], block_u_data[1201], block_u_data[1202], block_u_data[1203], block_u_data[1204], block_u_data[1205], block_u_data[1206], block_u_data[1207], block_u_data[1208], block_u_data[1209], block_u_data[1210], block_u_data[1211], block_u_data[1212], block_u_data[1213], block_u_data[1214], block_u_data[1215], block_u_data[1216], block_u_data[1217], block_u_data[1218], block_u_data[1219], block_u_data[1220], block_u_data[1221], block_u_data[1222], block_u_data[1223], block_u_data[1224], block_u_data[1225], block_u_data[1226], block_u_data[1227], block_u_data[1228], block_u_data[1229], block_u_data[1230], block_u_data[1231], block_u_data[1232], block_u_data[1233], block_u_data[1234], block_u_data[1235], block_u_data[1236], block_u_data[1237], block_u_data[1238], block_u_data[1239], block_u_data[1240], block_u_data[1241], block_u_data[1242], block_u_data[1243], block_u_data[1244], block_u_data[1245], block_u_data[1246], block_u_data[1247], block_u_data[1248], block_u_data[1249], block_u_data[1250], block_u_data[1251], block_u_data[1252], block_u_data[1253], block_u_data[1254], block_u_data[1255], block_u_data[1256], block_u_data[1257], block_u_data[1258], block_u_data[1259], block_u_data[1260], block_u_data[1261], block_u_data[1262], block_u_data[1263], block_u_data[1264], block_u_data[1265], block_u_data[1266], block_u_data[1267], block_u_data[1268], block_u_data[1269], block_u_data[1270], block_u_data[1271], block_u_data[1272], block_u_data[1273], block_u_data[1274], block_u_data[1275], block_u_data[1276], block_u_data[1277], block_u_data[1278], block_u_data[1279], block_u_data[1280], block_u_data[1281], block_u_data[1282], block_u_data[1283], block_u_data[1284], block_u_data[1285], block_u_data[1286], block_u_data[1287], block_u_data[1288], block_u_data[1289], block_u_data[1290], block_u_data[1291], block_u_data[1292], block_u_data[1293], block_u_data[1294], block_u_data[1295], block_u_data[1296], block_u_data[1297], block_u_data[1298], block_u_data[1299], block_u_data[1300], block_u_data[1301], block_u_data[1302], block_u_data[1303], block_u_data[1304], block_u_data[1305], block_u_data[1306], block_u_data[1307], block_u_data[1308], block_u_data[1309], block_u_data[1310], block_u_data[1311], block_u_data[1312], block_u_data[1313], block_u_data[1314], block_u_data[1315], block_u_data[1316], block_u_data[1317], block_u_data[1318], block_u_data[1319], block_u_data[1320], block_u_data[1321], block_u_data[1322], block_u_data[1323], block_u_data[1324], block_u_data[1325], block_u_data[1326], block_u_data[1327], block_u_data[1328], block_u_data[1329], block_u_data[1330], block_u_data[1331], block_u_data[1332], block_u_data[1333], block_u_data[1334], block_u_data[1335], block_u_data[1336], block_u_data[1337], block_u_data[1338], block_u_data[1339], block_u_data[1340], block_u_data[1341], block_u_data[1342], block_u_data[1343], block_u_data[1344], block_u_data[1345], block_u_data[1346], block_u_data[1347], block_u_data[1348], block_u_data[1349], block_u_data[1350], block_u_data[1351], block_u_data[1352], block_u_data[1353], block_u_data[1354], block_u_data[1355], block_u_data[1356], block_u_data[1357], block_u_data[1358], block_u_data[1359], block_u_data[1360], block_u_data[1361], block_u_data[1362], block_u_data[1363], block_u_data[1364], block_u_data[1365], block_u_data[1366], block_u_data[1367], block_u_data[1368], block_u_data[1369], block_u_data[1370], block_u_data[1371], block_u_data[1372], block_u_data[1373], block_u_data[1374], block_u_data[1375], block_u_data[1376], block_u_data[1377], block_u_data[1378], block_u_data[1379], block_u_data[1380], block_u_data[1381], block_u_data[1382], block_u_data[1383], block_u_data[1384], block_u_data[1385], block_u_data[1386], block_u_data[1387], block_u_data[1388], block_u_data[1389], block_u_data[1390], block_u_data[1391], block_u_data[1392], block_u_data[1393], block_u_data[1394], block_u_data[1395], block_u_data[1396], block_u_data[1397], block_u_data[1398], block_u_data[1399], block_u_data[1400], block_u_data[1401], block_u_data[1402], block_u_data[1403], block_u_data[1404], block_u_data[1405], block_u_data[1406], block_u_data[1407], block_u_data[1408], block_u_data[1409], block_u_data[1410], block_u_data[1411], block_u_data[1412], block_u_data[1413], block_u_data[1414], block_u_data[1415], block_u_data[1416], block_u_data[1417], block_u_data[1418], block_u_data[1419], block_u_data[1420], block_u_data[1421], block_u_data[1422], block_u_data[1423], block_u_data[1424], block_u_data[1425], block_u_data[1426], block_u_data[1427], block_u_data[1428], block_u_data[1429], block_u_data[1430], block_u_data[1431], block_u_data[1432], block_u_data[1433], block_u_data[1434], block_u_data[1435], block_u_data[1436], block_u_data[1437], block_u_data[1438], block_u_data[1439], block_u_data[1440], block_u_data[1441], block_u_data[1442], block_u_data[1443], block_u_data[1444], block_u_data[1445], block_u_data[1446], block_u_data[1447], block_u_data[1448], block_u_data[1449], block_u_data[1450], block_u_data[1451], block_u_data[1452], block_u_data[1453], block_u_data[1454], block_u_data[1455], block_u_data[1456], block_u_data[1457], block_u_data[1458], block_u_data[1459], block_u_data[1460], block_u_data[1461], block_u_data[1462], block_u_data[1463], block_u_data[1464], block_u_data[1465], block_u_data[1466], block_u_data[1467], block_u_data[1468], block_u_data[1469], block_u_data[1470], block_u_data[1471], block_u_data[1472], block_u_data[1473], block_u_data[1474], block_u_data[1475], block_u_data[1476], block_u_data[1477], block_u_data[1478], block_u_data[1479], block_u_data[1480], block_u_data[1481], block_u_data[1482], block_u_data[1483], block_u_data[1484], block_u_data[1485], block_u_data[1486], block_u_data[1487], block_u_data[1488], block_u_data[1489], block_u_data[1490], block_u_data[1491], block_u_data[1492], block_u_data[1493], block_u_data[1494], block_u_data[1495], block_u_data[1496], block_u_data[1497], block_u_data[1498], block_u_data[1499], block_u_data[1500], block_u_data[1501], block_u_data[1502], block_u_data[1503], block_u_data[1504], block_u_data[1505], block_u_data[1506], block_u_data[1507], block_u_data[1508], block_u_data[1509], block_u_data[1510], block_u_data[1511], block_u_data[1512], block_u_data[1513], block_u_data[1514], block_u_data[1515], block_u_data[1516], block_u_data[1517], block_u_data[1518], block_u_data[1519], block_u_data[1520], block_u_data[1521], block_u_data[1522], block_u_data[1523], block_u_data[1524], block_u_data[1525], block_u_data[1526], block_u_data[1527], block_u_data[1528], block_u_data[1529], block_u_data[1530], block_u_data[1531], block_u_data[1532], block_u_data[1533], block_u_data[1534], block_u_data[1535], block_u_data[1536], block_u_data[1537], block_u_data[1538], block_u_data[1539], block_u_data[1540], block_u_data[1541], block_u_data[1542], block_u_data[1543], block_u_data[1544], block_u_data[1545], block_u_data[1546], block_u_data[1547], block_u_data[1548], block_u_data[1549], block_u_data[1550], block_u_data[1551], block_u_data[1552], block_u_data[1553], block_u_data[1554], block_u_data[1555], block_u_data[1556], block_u_data[1557], block_u_data[1558], block_u_data[1559], block_u_data[1560], block_u_data[1561], block_u_data[1562], block_u_data[1563], block_u_data[1564], block_u_data[1565], block_u_data[1566], block_u_data[1567], block_u_data[1568], block_u_data[1569], block_u_data[1570], block_u_data[1571], block_u_data[1572], block_u_data[1573], block_u_data[1574], block_u_data[1575], block_u_data[1576], block_u_data[1577], block_u_data[1578], block_u_data[1579], block_u_data[1580], block_u_data[1581], block_u_data[1582], block_u_data[1583], block_u_data[1584], block_u_data[1585], block_u_data[1586], block_u_data[1587], block_u_data[1588], block_u_data[1589], block_u_data[1590], block_u_data[1591], block_u_data[1592], block_u_data[1593], block_u_data[1594], block_u_data[1595], block_u_data[1596], block_u_data[1597], block_u_data[1598], block_u_data[1599], block_u_data[1600], block_u_data[1601], block_u_data[1602], block_u_data[1603], block_u_data[1604], block_u_data[1605], block_u_data[1606], block_u_data[1607], block_u_data[1608], block_u_data[1609], block_u_data[1610], block_u_data[1611], block_u_data[1612], block_u_data[1613], block_u_data[1614], block_u_data[1615], block_u_data[1616], block_u_data[1617], block_u_data[1618], block_u_data[1619], block_u_data[1620], block_u_data[1621], block_u_data[1622], block_u_data[1623], block_u_data[1624], block_u_data[1625], block_u_data[1626], block_u_data[1627], block_u_data[1628], block_u_data[1629], block_u_data[1630], block_u_data[1631], block_u_data[1632], block_u_data[1633], block_u_data[1634], block_u_data[1635], block_u_data[1636], block_u_data[1637], block_u_data[1638], block_u_data[1639], block_u_data[1640], block_u_data[1641], block_u_data[1642], block_u_data[1643], block_u_data[1644], block_u_data[1645], block_u_data[1646], block_u_data[1647], block_u_data[1648], block_u_data[1649], block_u_data[1650], block_u_data[1651], block_u_data[1652], block_u_data[1653], block_u_data[1654], block_u_data[1655], block_u_data[1656], block_u_data[1657], block_u_data[1658], block_u_data[1659], block_u_data[1660], block_u_data[1661], block_u_data[1662], block_u_data[1663], block_u_data[1664], block_u_data[1665], block_u_data[1666], block_u_data[1667], block_u_data[1668], block_u_data[1669], block_u_data[1670], block_u_data[1671], block_u_data[1672], block_u_data[1673], block_u_data[1674], block_u_data[1675], block_u_data[1676], block_u_data[1677], block_u_data[1678], block_u_data[1679], block_u_data[1680], block_u_data[1681], block_u_data[1682], block_u_data[1683], block_u_data[1684], block_u_data[1685], block_u_data[1686], block_u_data[1687], block_u_data[1688], block_u_data[1689], block_u_data[1690], block_u_data[1691], block_u_data[1692], block_u_data[1693], block_u_data[1694], block_u_data[1695], block_u_data[1696], block_u_data[1697], block_u_data[1698], block_u_data[1699], block_u_data[1700], block_u_data[1701], block_u_data[1702], block_u_data[1703], block_u_data[1704], block_u_data[1705], block_u_data[1706], block_u_data[1707], block_u_data[1708], block_u_data[1709], block_u_data[1710], block_u_data[1711], block_u_data[1712], block_u_data[1713], block_u_data[1714], block_u_data[1715], block_u_data[1716], block_u_data[1717], block_u_data[1718], block_u_data[1719], block_u_data[1720], block_u_data[1721], block_u_data[1722], block_u_data[1723], block_u_data[1724], block_u_data[1725], block_u_data[1726], block_u_data[1727], block_u_data[1728], block_u_data[1729], block_u_data[1730], block_u_data[1731], block_u_data[1732], block_u_data[1733], block_u_data[1734], block_u_data[1735], block_u_data[1736], block_u_data[1737], block_u_data[1738], block_u_data[1739], block_u_data[1740], block_u_data[1741], block_u_data[1742], block_u_data[1743], block_u_data[1744], block_u_data[1745], block_u_data[1746], block_u_data[1747], block_u_data[1748], block_u_data[1749], block_u_data[1750], block_u_data[1751], block_u_data[1752], block_u_data[1753], block_u_data[1754], block_u_data[1755], block_u_data[1756], block_u_data[1757], block_u_data[1758], block_u_data[1759], block_u_data[1760], block_u_data[1761], block_u_data[1762], block_u_data[1763], block_u_data[1764], block_u_data[1765], block_u_data[1766], block_u_data[1767], block_u_data[1768], block_u_data[1769], block_u_data[1770], block_u_data[1771], block_u_data[1772], block_u_data[1773], block_u_data[1774], block_u_data[1775], block_u_data[1776], block_u_data[1777], block_u_data[1778], block_u_data[1779], block_u_data[1780], block_u_data[1781], block_u_data[1782], block_u_data[1783], block_u_data[1784], block_u_data[1785], block_u_data[1786], block_u_data[1787], block_u_data[1788], block_u_data[1789], block_u_data[1790], block_u_data[1791], block_u_data[1792], block_u_data[1793], block_u_data[1794], block_u_data[1795], block_u_data[1796], block_u_data[1797], block_u_data[1798], block_u_data[1799], block_u_data[1800], block_u_data[1801], block_u_data[1802], block_u_data[1803], block_u_data[1804], block_u_data[1805], block_u_data[1806], block_u_data[1807], block_u_data[1808], block_u_data[1809], block_u_data[1810], block_u_data[1811], block_u_data[1812], block_u_data[1813], block_u_data[1814], block_u_data[1815], block_u_data[1816], block_u_data[1817], block_u_data[1818], block_u_data[1819], block_u_data[1820], block_u_data[1821], block_u_data[1822], block_u_data[1823], block_u_data[1824], block_u_data[1825], block_u_data[1826], block_u_data[1827], block_u_data[1828], block_u_data[1829], block_u_data[1830], block_u_data[1831], block_u_data[1832], block_u_data[1833], block_u_data[1834], block_u_data[1835], block_u_data[1836], block_u_data[1837], block_u_data[1838], block_u_data[1839], block_u_data[1840], block_u_data[1841], block_u_data[1842], block_u_data[1843], block_u_data[1844], block_u_data[1845], block_u_data[1846], block_u_data[1847], block_u_data[1848], block_u_data[1849], block_u_data[1850], block_u_data[1851], block_u_data[1852], block_u_data[1853], block_u_data[1854], block_u_data[1855], block_u_data[1856], block_u_data[1857], block_u_data[1858], block_u_data[1859], block_u_data[1860], block_u_data[1861], block_u_data[1862], block_u_data[1863], block_u_data[1864], block_u_data[1865], block_u_data[1866], block_u_data[1867], block_u_data[1868], block_u_data[1869], block_u_data[1870], block_u_data[1871], block_u_data[1872], block_u_data[1873], block_u_data[1874], block_u_data[1875], block_u_data[1876], block_u_data[1877], block_u_data[1878], block_u_data[1879], block_u_data[1880], block_u_data[1881], block_u_data[1882], block_u_data[1883], block_u_data[1884], block_u_data[1885], block_u_data[1886], block_u_data[1887], block_u_data[1888], block_u_data[1889], block_u_data[1890], block_u_data[1891], block_u_data[1892], block_u_data[1893], block_u_data[1894], block_u_data[1895], block_u_data[1896], block_u_data[1897], block_u_data[1898], block_u_data[1899], block_u_data[1900], block_u_data[1901], block_u_data[1902], block_u_data[1903], block_u_data[1904], block_u_data[1905], block_u_data[1906], block_u_data[1907], block_u_data[1908], block_u_data[1909], block_u_data[1910], block_u_data[1911], block_u_data[1912], block_u_data[1913], block_u_data[1914], block_u_data[1915], block_u_data[1916], block_u_data[1917], block_u_data[1918], block_u_data[1919], block_u_data[1920], block_u_data[1921], block_u_data[1922], block_u_data[1923], block_u_data[1924], block_u_data[1925], block_u_data[1926], block_u_data[1927], block_u_data[1928], block_u_data[1929], block_u_data[1930], block_u_data[1931], block_u_data[1932], block_u_data[1933], block_u_data[1934], block_u_data[1935], block_u_data[1936], block_u_data[1937], block_u_data[1938], block_u_data[1939], block_u_data[1940], block_u_data[1941], block_u_data[1942], block_u_data[1943], block_u_data[1944], block_u_data[1945], block_u_data[1946], block_u_data[1947], block_u_data[1948], block_u_data[1949], block_u_data[1950], block_u_data[1951], block_u_data[1952], block_u_data[1953], block_u_data[1954], block_u_data[1955], block_u_data[1956], block_u_data[1957], block_u_data[1958], block_u_data[1959], block_u_data[1960], block_u_data[1961], block_u_data[1962], block_u_data[1963], block_u_data[1964], block_u_data[1965], block_u_data[1966], block_u_data[1967], block_u_data[1968], block_u_data[1969], block_u_data[1970], block_u_data[1971], block_u_data[1972], block_u_data[1973], block_u_data[1974], block_u_data[1975], block_u_data[1976], block_u_data[1977], block_u_data[1978], block_u_data[1979], block_u_data[1980], block_u_data[1981], block_u_data[1982], block_u_data[1983], block_u_data[1984], block_u_data[1985], block_u_data[1986], block_u_data[1987], block_u_data[1988], block_u_data[1989], block_u_data[1990], block_u_data[1991], block_u_data[1992], block_u_data[1993], block_u_data[1994], block_u_data[1995], block_u_data[1996], block_u_data[1997], block_u_data[1998], block_u_data[1999], block_u_data[2000], block_u_data[2001], block_u_data[2002], block_u_data[2003], block_u_data[2004], block_u_data[2005], block_u_data[2006], block_u_data[2007], block_u_data[2008], block_u_data[2009], block_u_data[2010], block_u_data[2011], block_u_data[2012], block_u_data[2013], block_u_data[2014], block_u_data[2015], block_u_data[2016], block_u_data[2017], block_u_data[2018], block_u_data[2019], block_u_data[2020], block_u_data[2021], block_u_data[2022], block_u_data[2023], block_u_data[2024], block_u_data[2025], block_u_data[2026], block_u_data[2027], block_u_data[2028], block_u_data[2029], block_u_data[2030], block_u_data[2031], block_u_data[2032], block_u_data[2033], block_u_data[2034], block_u_data[2035], block_u_data[2036], block_u_data[2037], block_u_data[2038], block_u_data[2039], block_u_data[2040], block_u_data[2041], block_u_data[2042], block_u_data[2043], block_u_data[2044], block_u_data[2045], block_u_data[2046], block_u_data[2047], block_u_data[2048], block_u_data[2049], block_u_data[2050], block_u_data[2051], block_u_data[2052], block_u_data[2053], block_u_data[2054], block_u_data[2055], block_u_data[2056], block_u_data[2057], block_u_data[2058], block_u_data[2059], block_u_data[2060], block_u_data[2061], block_u_data[2062], block_u_data[2063], block_u_data[2064], block_u_data[2065], block_u_data[2066], block_u_data[2067], block_u_data[2068], block_u_data[2069], block_u_data[2070], block_u_data[2071], block_u_data[2072], block_u_data[2073], block_u_data[2074], block_u_data[2075], block_u_data[2076], block_u_data[2077], block_u_data[2078], block_u_data[2079], block_u_data[2080], block_u_data[2081], block_u_data[2082], block_u_data[2083], block_u_data[2084], block_u_data[2085], block_u_data[2086], block_u_data[2087], block_u_data[2088], block_u_data[2089], block_u_data[2090], block_u_data[2091], block_u_data[2092], block_u_data[2093], block_u_data[2094], block_u_data[2095], block_u_data[2096], block_u_data[2097], block_u_data[2098], block_u_data[2099], block_u_data[2100], block_u_data[2101], block_u_data[2102], block_u_data[2103], block_u_data[2104], block_u_data[2105], block_u_data[2106], block_u_data[2107], block_u_data[2108], block_u_data[2109], block_u_data[2110], block_u_data[2111], block_u_data[2112], block_u_data[2113], block_u_data[2114], block_u_data[2115], block_u_data[2116], block_u_data[2117], block_u_data[2118], block_u_data[2119], block_u_data[2120], block_u_data[2121], block_u_data[2122], block_u_data[2123], block_u_data[2124], block_u_data[2125], block_u_data[2126], block_u_data[2127], block_u_data[2128], block_u_data[2129], block_u_data[2130], block_u_data[2131], block_u_data[2132], block_u_data[2133], block_u_data[2134], block_u_data[2135], block_u_data[2136], block_u_data[2137], block_u_data[2138], block_u_data[2139], block_u_data[2140], block_u_data[2141], block_u_data[2142], block_u_data[2143], block_u_data[2144], block_u_data[2145], block_u_data[2146], block_u_data[2147], block_u_data[2148], block_u_data[2149], block_u_data[2150], block_u_data[2151], block_u_data[2152], block_u_data[2153], block_u_data[2154], block_u_data[2155], block_u_data[2156], block_u_data[2157], block_u_data[2158], block_u_data[2159], block_u_data[2160], block_u_data[2161], block_u_data[2162], block_u_data[2163], block_u_data[2164], block_u_data[2165], block_u_data[2166], block_u_data[2167], block_u_data[2168], block_u_data[2169], block_u_data[2170], block_u_data[2171], block_u_data[2172], block_u_data[2173], block_u_data[2174], block_u_data[2175], block_u_data[2176], block_u_data[2177], block_u_data[2178], block_u_data[2179], block_u_data[2180], block_u_data[2181], block_u_data[2182], block_u_data[2183], block_u_data[2184], block_u_data[2185], block_u_data[2186]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    35,
    76,
    1,
    0,
    0,
    0,
    1,
    0,
    6,
    31,
    139,
    921430,
    1357710,
    22,
    16,
    10,
    180,
    10,
    2,
    2187,
    0,
    2187,
    2187,
    2187,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl(
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
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
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

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_packed_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
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

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_packed_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
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

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    190,
    387,
    1,
    0,
    10,
    0,
    1,
    0,
    6,
    129,
    615,
    921430,
    1357710,
    120,
    69,
    10,
    180,
    10,
    2,
    2187,
    2187,
    2187,
    2187,
    2187,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl(
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
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
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

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_packed_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
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

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_packed_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
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

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
