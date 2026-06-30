#include "../neohookean_ogden_d3_tensor_product_local.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

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
    43,
    81,
    1,
    0,
    10,
    0,
    1,
    0,
    9,
    30,
    162,
    20,
    23,
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_adjugate4,
        const double *const SFEM_RESTRICT g_jacobian_adjugate5,
        const double *const SFEM_RESTRICT g_jacobian_adjugate6,
        const double *const SFEM_RESTRICT g_jacobian_adjugate7,
        const double *const SFEM_RESTRICT g_jacobian_adjugate8,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_adjugate4,
        const float *const SFEM_RESTRICT g_jacobian_adjugate5,
        const float *const SFEM_RESTRICT g_jacobian_adjugate6,
        const float *const SFEM_RESTRICT g_jacobian_adjugate7,
        const float *const SFEM_RESTRICT g_jacobian_adjugate8,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_adjugate4,
        const double *const SFEM_RESTRICT g_jacobian_adjugate5,
        const double *const SFEM_RESTRICT g_jacobian_adjugate6,
        const double *const SFEM_RESTRICT g_jacobian_adjugate7,
        const double *const SFEM_RESTRICT g_jacobian_adjugate8,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_adjugate4,
        const float *const SFEM_RESTRICT g_jacobian_adjugate5,
        const float *const SFEM_RESTRICT g_jacobian_adjugate6,
        const float *const SFEM_RESTRICT g_jacobian_adjugate7,
        const float *const SFEM_RESTRICT g_jacobian_adjugate8,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
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
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        const scalar_t *block_coordinate_streams[DIM * N_SHAPE];
        for (int stream = 0; stream < DIM * N_SHAPE; ++stream) {
            block_coordinate_streams[stream] = block_coordinate_data[stream];
        }
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
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
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];
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
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        const scalar_t *block_coordinate_streams[DIM * N_SHAPE];
        for (int stream = 0; stream < DIM * N_SHAPE; ++stream) {
            block_coordinate_streams[stream] = block_coordinate_data[stream];
        }
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
        const double mu,
        const double lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}


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
    43,
    81,
    1,
    0,
    10,
    0,
    1,
    0,
    9,
    30,
    162,
    20,
    23,
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape] * out_stride;
                    #pragma omp atomic update
                    out_components[d][node] += block_out_data[shape * DIM + d][lane];
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
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_adjugate4,
        const double *const SFEM_RESTRICT g_jacobian_adjugate5,
        const double *const SFEM_RESTRICT g_jacobian_adjugate6,
        const double *const SFEM_RESTRICT g_jacobian_adjugate7,
        const double *const SFEM_RESTRICT g_jacobian_adjugate8,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_adjugate4,
        const float *const SFEM_RESTRICT g_jacobian_adjugate5,
        const float *const SFEM_RESTRICT g_jacobian_adjugate6,
        const float *const SFEM_RESTRICT g_jacobian_adjugate7,
        const float *const SFEM_RESTRICT g_jacobian_adjugate8,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
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
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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

        const scalar_t *block_coordinate_streams[DIM * N_SHAPE];
        for (int stream = 0; stream < DIM * N_SHAPE; ++stream) {
            block_coordinate_streams[stream] = block_coordinate_data[stream];
        }
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape] * out_stride;
                    #pragma omp atomic update
                    out_components[d][node] += block_out_data[shape * DIM + d][lane];
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
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


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
    216,
    404,
    1,
    0,
    20,
    0,
    1,
    0,
    9,
    136,
    668,
    117,
    103,
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
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
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape] * out_stride;
                    #pragma omp atomic update
                    out_components[d][node] += block_out_data[shape * DIM + d][lane];
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
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_adjugate4,
        const double *const SFEM_RESTRICT g_jacobian_adjugate5,
        const double *const SFEM_RESTRICT g_jacobian_adjugate6,
        const double *const SFEM_RESTRICT g_jacobian_adjugate7,
        const double *const SFEM_RESTRICT g_jacobian_adjugate8,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_adjugate4,
        const float *const SFEM_RESTRICT g_jacobian_adjugate5,
        const float *const SFEM_RESTRICT g_jacobian_adjugate6,
        const float *const SFEM_RESTRICT g_jacobian_adjugate7,
        const float *const SFEM_RESTRICT g_jacobian_adjugate8,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
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
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
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
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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

        const scalar_t *block_coordinate_streams[DIM * N_SHAPE];
        for (int stream = 0; stream < DIM * N_SHAPE; ++stream) {
            block_coordinate_streams[stream] = block_coordinate_data[stream];
        }
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const int stream_shape = shape;
            for (int d = 0; d < DIM; ++d) {
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[lane * N_SHAPE + stream_shape] * out_stride;
                    #pragma omp atomic update
                    out_components[d][node] += block_out_data[shape * DIM + d][lane];
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
        const double mu,
        const double lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
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
    return sfem::codegen::neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

