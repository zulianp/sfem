#include <cstdio>
#include <type_traits>
#include "../laplace_d3_tensor_product_local.hpp"
#include "../laplace_d3_tensor_product_hessian.hpp"
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

static const KernelDiagnostics laplace_proteus_hex729_objective_soa_diagnostics_data = {
    "laplace_proteus_hex729_objective_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    2,
    2,
    0,
    0,
    3,
    0,
    0,
    0,
    3,
    1,
    7,
    453280,
    889560,
    0,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_objective_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_objective_soa",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_objective_soa_float",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_objective_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_objective_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_objective_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex729_objective_steps_affine_mesh_soa_impl(
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
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
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
                for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_value);

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

extern "C" int laplace_proteus_hex729_objective_steps_affine_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_proteus_hex729_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_proteus_hex729_objective_steps_affine_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_proteus_hex729_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_objective_steps_packed_affine_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_value);

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

extern "C" int laplace_proteus_hex729_objective_steps_packed_affine_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_value);

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
static SFEM_INLINE int laplace_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_value);

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

extern "C" int laplace_proteus_hex729_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_proteus_hex729_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_proteus_hex729_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                }
            }
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_value);

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

extern "C" int laplace_proteus_hex729_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                }
            }
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80], block_u_data[81], block_u_data[82], block_u_data[83], block_u_data[84], block_u_data[85], block_u_data[86], block_u_data[87], block_u_data[88], block_u_data[89], block_u_data[90], block_u_data[91], block_u_data[92], block_u_data[93], block_u_data[94], block_u_data[95], block_u_data[96], block_u_data[97], block_u_data[98], block_u_data[99], block_u_data[100], block_u_data[101], block_u_data[102], block_u_data[103], block_u_data[104], block_u_data[105], block_u_data[106], block_u_data[107], block_u_data[108], block_u_data[109], block_u_data[110], block_u_data[111], block_u_data[112], block_u_data[113], block_u_data[114], block_u_data[115], block_u_data[116], block_u_data[117], block_u_data[118], block_u_data[119], block_u_data[120], block_u_data[121], block_u_data[122], block_u_data[123], block_u_data[124], block_u_data[125], block_u_data[126], block_u_data[127], block_u_data[128], block_u_data[129], block_u_data[130], block_u_data[131], block_u_data[132], block_u_data[133], block_u_data[134], block_u_data[135], block_u_data[136], block_u_data[137], block_u_data[138], block_u_data[139], block_u_data[140], block_u_data[141], block_u_data[142], block_u_data[143], block_u_data[144], block_u_data[145], block_u_data[146], block_u_data[147], block_u_data[148], block_u_data[149], block_u_data[150], block_u_data[151], block_u_data[152], block_u_data[153], block_u_data[154], block_u_data[155], block_u_data[156], block_u_data[157], block_u_data[158], block_u_data[159], block_u_data[160], block_u_data[161], block_u_data[162], block_u_data[163], block_u_data[164], block_u_data[165], block_u_data[166], block_u_data[167], block_u_data[168], block_u_data[169], block_u_data[170], block_u_data[171], block_u_data[172], block_u_data[173], block_u_data[174], block_u_data[175], block_u_data[176], block_u_data[177], block_u_data[178], block_u_data[179], block_u_data[180], block_u_data[181], block_u_data[182], block_u_data[183], block_u_data[184], block_u_data[185], block_u_data[186], block_u_data[187], block_u_data[188], block_u_data[189], block_u_data[190], block_u_data[191], block_u_data[192], block_u_data[193], block_u_data[194], block_u_data[195], block_u_data[196], block_u_data[197], block_u_data[198], block_u_data[199], block_u_data[200], block_u_data[201], block_u_data[202], block_u_data[203], block_u_data[204], block_u_data[205], block_u_data[206], block_u_data[207], block_u_data[208], block_u_data[209], block_u_data[210], block_u_data[211], block_u_data[212], block_u_data[213], block_u_data[214], block_u_data[215], block_u_data[216], block_u_data[217], block_u_data[218], block_u_data[219], block_u_data[220], block_u_data[221], block_u_data[222], block_u_data[223], block_u_data[224], block_u_data[225], block_u_data[226], block_u_data[227], block_u_data[228], block_u_data[229], block_u_data[230], block_u_data[231], block_u_data[232], block_u_data[233], block_u_data[234], block_u_data[235], block_u_data[236], block_u_data[237], block_u_data[238], block_u_data[239], block_u_data[240], block_u_data[241], block_u_data[242], block_u_data[243], block_u_data[244], block_u_data[245], block_u_data[246], block_u_data[247], block_u_data[248], block_u_data[249], block_u_data[250], block_u_data[251], block_u_data[252], block_u_data[253], block_u_data[254], block_u_data[255], block_u_data[256], block_u_data[257], block_u_data[258], block_u_data[259], block_u_data[260], block_u_data[261], block_u_data[262], block_u_data[263], block_u_data[264], block_u_data[265], block_u_data[266], block_u_data[267], block_u_data[268], block_u_data[269], block_u_data[270], block_u_data[271], block_u_data[272], block_u_data[273], block_u_data[274], block_u_data[275], block_u_data[276], block_u_data[277], block_u_data[278], block_u_data[279], block_u_data[280], block_u_data[281], block_u_data[282], block_u_data[283], block_u_data[284], block_u_data[285], block_u_data[286], block_u_data[287], block_u_data[288], block_u_data[289], block_u_data[290], block_u_data[291], block_u_data[292], block_u_data[293], block_u_data[294], block_u_data[295], block_u_data[296], block_u_data[297], block_u_data[298], block_u_data[299], block_u_data[300], block_u_data[301], block_u_data[302], block_u_data[303], block_u_data[304], block_u_data[305], block_u_data[306], block_u_data[307], block_u_data[308], block_u_data[309], block_u_data[310], block_u_data[311], block_u_data[312], block_u_data[313], block_u_data[314], block_u_data[315], block_u_data[316], block_u_data[317], block_u_data[318], block_u_data[319], block_u_data[320], block_u_data[321], block_u_data[322], block_u_data[323], block_u_data[324], block_u_data[325], block_u_data[326], block_u_data[327], block_u_data[328], block_u_data[329], block_u_data[330], block_u_data[331], block_u_data[332], block_u_data[333], block_u_data[334], block_u_data[335], block_u_data[336], block_u_data[337], block_u_data[338], block_u_data[339], block_u_data[340], block_u_data[341], block_u_data[342], block_u_data[343], block_u_data[344], block_u_data[345], block_u_data[346], block_u_data[347], block_u_data[348], block_u_data[349], block_u_data[350], block_u_data[351], block_u_data[352], block_u_data[353], block_u_data[354], block_u_data[355], block_u_data[356], block_u_data[357], block_u_data[358], block_u_data[359], block_u_data[360], block_u_data[361], block_u_data[362], block_u_data[363], block_u_data[364], block_u_data[365], block_u_data[366], block_u_data[367], block_u_data[368], block_u_data[369], block_u_data[370], block_u_data[371], block_u_data[372], block_u_data[373], block_u_data[374], block_u_data[375], block_u_data[376], block_u_data[377], block_u_data[378], block_u_data[379], block_u_data[380], block_u_data[381], block_u_data[382], block_u_data[383], block_u_data[384], block_u_data[385], block_u_data[386], block_u_data[387], block_u_data[388], block_u_data[389], block_u_data[390], block_u_data[391], block_u_data[392], block_u_data[393], block_u_data[394], block_u_data[395], block_u_data[396], block_u_data[397], block_u_data[398], block_u_data[399], block_u_data[400], block_u_data[401], block_u_data[402], block_u_data[403], block_u_data[404], block_u_data[405], block_u_data[406], block_u_data[407], block_u_data[408], block_u_data[409], block_u_data[410], block_u_data[411], block_u_data[412], block_u_data[413], block_u_data[414], block_u_data[415], block_u_data[416], block_u_data[417], block_u_data[418], block_u_data[419], block_u_data[420], block_u_data[421], block_u_data[422], block_u_data[423], block_u_data[424], block_u_data[425], block_u_data[426], block_u_data[427], block_u_data[428], block_u_data[429], block_u_data[430], block_u_data[431], block_u_data[432], block_u_data[433], block_u_data[434], block_u_data[435], block_u_data[436], block_u_data[437], block_u_data[438], block_u_data[439], block_u_data[440], block_u_data[441], block_u_data[442], block_u_data[443], block_u_data[444], block_u_data[445], block_u_data[446], block_u_data[447], block_u_data[448], block_u_data[449], block_u_data[450], block_u_data[451], block_u_data[452], block_u_data[453], block_u_data[454], block_u_data[455], block_u_data[456], block_u_data[457], block_u_data[458], block_u_data[459], block_u_data[460], block_u_data[461], block_u_data[462], block_u_data[463], block_u_data[464], block_u_data[465], block_u_data[466], block_u_data[467], block_u_data[468], block_u_data[469], block_u_data[470], block_u_data[471], block_u_data[472], block_u_data[473], block_u_data[474], block_u_data[475], block_u_data[476], block_u_data[477], block_u_data[478], block_u_data[479], block_u_data[480], block_u_data[481], block_u_data[482], block_u_data[483], block_u_data[484], block_u_data[485], block_u_data[486], block_u_data[487], block_u_data[488], block_u_data[489], block_u_data[490], block_u_data[491], block_u_data[492], block_u_data[493], block_u_data[494], block_u_data[495], block_u_data[496], block_u_data[497], block_u_data[498], block_u_data[499], block_u_data[500], block_u_data[501], block_u_data[502], block_u_data[503], block_u_data[504], block_u_data[505], block_u_data[506], block_u_data[507], block_u_data[508], block_u_data[509], block_u_data[510], block_u_data[511], block_u_data[512], block_u_data[513], block_u_data[514], block_u_data[515], block_u_data[516], block_u_data[517], block_u_data[518], block_u_data[519], block_u_data[520], block_u_data[521], block_u_data[522], block_u_data[523], block_u_data[524], block_u_data[525], block_u_data[526], block_u_data[527], block_u_data[528], block_u_data[529], block_u_data[530], block_u_data[531], block_u_data[532], block_u_data[533], block_u_data[534], block_u_data[535], block_u_data[536], block_u_data[537], block_u_data[538], block_u_data[539], block_u_data[540], block_u_data[541], block_u_data[542], block_u_data[543], block_u_data[544], block_u_data[545], block_u_data[546], block_u_data[547], block_u_data[548], block_u_data[549], block_u_data[550], block_u_data[551], block_u_data[552], block_u_data[553], block_u_data[554], block_u_data[555], block_u_data[556], block_u_data[557], block_u_data[558], block_u_data[559], block_u_data[560], block_u_data[561], block_u_data[562], block_u_data[563], block_u_data[564], block_u_data[565], block_u_data[566], block_u_data[567], block_u_data[568], block_u_data[569], block_u_data[570], block_u_data[571], block_u_data[572], block_u_data[573], block_u_data[574], block_u_data[575], block_u_data[576], block_u_data[577], block_u_data[578], block_u_data[579], block_u_data[580], block_u_data[581], block_u_data[582], block_u_data[583], block_u_data[584], block_u_data[585], block_u_data[586], block_u_data[587], block_u_data[588], block_u_data[589], block_u_data[590], block_u_data[591], block_u_data[592], block_u_data[593], block_u_data[594], block_u_data[595], block_u_data[596], block_u_data[597], block_u_data[598], block_u_data[599], block_u_data[600], block_u_data[601], block_u_data[602], block_u_data[603], block_u_data[604], block_u_data[605], block_u_data[606], block_u_data[607], block_u_data[608], block_u_data[609], block_u_data[610], block_u_data[611], block_u_data[612], block_u_data[613], block_u_data[614], block_u_data[615], block_u_data[616], block_u_data[617], block_u_data[618], block_u_data[619], block_u_data[620], block_u_data[621], block_u_data[622], block_u_data[623], block_u_data[624], block_u_data[625], block_u_data[626], block_u_data[627], block_u_data[628], block_u_data[629], block_u_data[630], block_u_data[631], block_u_data[632], block_u_data[633], block_u_data[634], block_u_data[635], block_u_data[636], block_u_data[637], block_u_data[638], block_u_data[639], block_u_data[640], block_u_data[641], block_u_data[642], block_u_data[643], block_u_data[644], block_u_data[645], block_u_data[646], block_u_data[647], block_u_data[648], block_u_data[649], block_u_data[650], block_u_data[651], block_u_data[652], block_u_data[653], block_u_data[654], block_u_data[655], block_u_data[656], block_u_data[657], block_u_data[658], block_u_data[659], block_u_data[660], block_u_data[661], block_u_data[662], block_u_data[663], block_u_data[664], block_u_data[665], block_u_data[666], block_u_data[667], block_u_data[668], block_u_data[669], block_u_data[670], block_u_data[671], block_u_data[672], block_u_data[673], block_u_data[674], block_u_data[675], block_u_data[676], block_u_data[677], block_u_data[678], block_u_data[679], block_u_data[680], block_u_data[681], block_u_data[682], block_u_data[683], block_u_data[684], block_u_data[685], block_u_data[686], block_u_data[687], block_u_data[688], block_u_data[689], block_u_data[690], block_u_data[691], block_u_data[692], block_u_data[693], block_u_data[694], block_u_data[695], block_u_data[696], block_u_data[697], block_u_data[698], block_u_data[699], block_u_data[700], block_u_data[701], block_u_data[702], block_u_data[703], block_u_data[704], block_u_data[705], block_u_data[706], block_u_data[707], block_u_data[708], block_u_data[709], block_u_data[710], block_u_data[711], block_u_data[712], block_u_data[713], block_u_data[714], block_u_data[715], block_u_data[716], block_u_data[717], block_u_data[718], block_u_data[719], block_u_data[720], block_u_data[721], block_u_data[722], block_u_data[723], block_u_data[724], block_u_data[725], block_u_data[726], block_u_data[727], block_u_data[728]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    laplace_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_value);

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

static const KernelDiagnostics laplace_proteus_hex729_gradient_soa_diagnostics_data = {
    "laplace_proteus_hex729_gradient_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    0,
    3,
    0,
    0,
    0,
    0,
    0,
    0,
    3,
    3,
    3,
    921430,
    1357710,
    0,
    2,
    10,
    180,
    10,
    2,
    2187,
    0,
    729,
    729,
    729,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_gradient_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_gradient_soa",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_gradient_soa_float",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_gradient_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_gradient_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_gradient_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex729_gradient_affine_mesh_soa_impl(
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
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
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

        laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex729_gradient_affine_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_proteus_hex729_gradient_affine_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, u_stride, ux, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_gradient_packed_affine_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_gradient_packed_affine_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_gradient_packed_two_pass_affine_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_gradient_packed_two_pass_affine_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int laplace_proteus_hex729_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex729_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_proteus_hex729_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex729_apply_soa_diagnostics_data = {
    "laplace_proteus_hex729_apply_soa",
    "PROTEUS_HEX729",
    3,
    1000,
    729,
    16,
    10,
    0,
    3,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    3,
    3,
    921430,
    1357710,
    0,
    2,
    10,
    180,
    10,
    2,
    0,
    2187,
    729,
    729,
    729,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_apply_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex729_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex729_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_apply_soa",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex729_apply_soa_float",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_apply_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex729_apply_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex729_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_apply_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex729_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex729_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex729_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex729_apply_affine_mesh_soa_impl(
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
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
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

        laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_h_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex729_apply_affine_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_proteus_hex729_apply_affine_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, h_stride, hx, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_apply_packed_affine_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_apply_packed_affine_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_apply_packed_two_pass_affine_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_apply_packed_two_pass_affine_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 729;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex729_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 9;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<geom_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<geom_t, scalar_t>());

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int laplace_proteus_hex729_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_h_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex729_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_proteus_hex729_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_proteus_hex729_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_proteus_hex729_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int laplace_proteus_hex729_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex729_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 10;
    static constexpr int N_SHAPE_1D = 9;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                laplace_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static SFEM_INLINE void laplace_proteus_hex729_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
    for (int d = 0; d < 729; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
        for (int d = 0; d < 729; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_FIELD_COMPONENTS = 3;
    static constexpr int N_SHAPE = 729;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        laplace_proteus_hex729_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            entries[i * N_SHAPE + j] = row_begin + ks[j];
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            scalar_t *const block = &values[entries[i * N_SHAPE + j] * N_FIELD_COMPONENTS * N_FIELD_COMPONENTS];
            for (int bi = 0; bi < N_FIELD_COMPONENTS; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < N_FIELD_COMPONENTS; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * N_FIELD_COMPONENTS + bj] += element_matrix[row * (N_FIELD_COMPONENTS * N_SHAPE) + col];
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_SHAPE = 729;
    count_t row_begin[N_SHAPE];
    int lenrow[N_SHAPE];
    int local_col[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        row_begin[i] = rowptr[ev[i]];
        lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin[i]];
        laplace_proteus_hex729_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow[i], ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            local_col[i * N_SHAPE + j] = (int)ks[j];
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t rb = row_begin[i];
        const int lr = lenrow[i];
        for (int j = 0; j < N_SHAPE; ++j) {
            const int lc = local_col[i * N_SHAPE + j];
            for (int bi = 0; bi < N_FIELD_COMPONENTS; ++bi) {
                const int row = bi * N_SHAPE + i;
                scalar_t *const row_values = &values[rb * N_FIELD_COMPONENTS * N_FIELD_COMPONENTS + bi * lr * N_FIELD_COMPONENTS];
                for (int bj = 0; bj < N_FIELD_COMPONENTS; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    row_values[lc * N_FIELD_COMPONENTS + bj] += element_matrix[row * (N_FIELD_COMPONENTS * N_SHAPE) + col];
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_dia(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t nnodes,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_SHAPE = 729;
    ptrdiff_t diagonals[N_SHAPE * N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            const int offset = (int)(ev[j] - ev[i]);
            ptrdiff_t diagonal = 0;
            while (diagonal < ndiag && diag_offsets[diagonal] != offset) ++diagonal;
            diagonals[i * N_SHAPE + j] = diagonal;
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            const ptrdiff_t diagonal = diagonals[i * N_SHAPE + j];
            scalar_t *const block = &values[(diagonal * nnodes + ev[i]) * N_FIELD_COMPONENTS * N_FIELD_COMPONENTS];
            for (int bi = 0; bi < N_FIELD_COMPONENTS; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < N_FIELD_COMPONENTS; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * N_FIELD_COMPONENTS + bj] += element_matrix[row * (N_FIELD_COMPONENTS * N_SHAPE) + col];
                }
            }
        }
    }
}

template <typename scalar_t, typename geometry_t, int FORMAT>
static int laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        const ptrdiff_t coo_nnz,
        const idx_t *const SFEM_RESTRICT coo_rows,
        const idx_t *const SFEM_RESTRICT coo_cols,
        idx_t *const SFEM_RESTRICT coo_triplet_rows,
        idx_t *const SFEM_RESTRICT coo_triplet_cols) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1000;
    static constexpr int N_SHAPE = 729;
    static constexpr int VECTOR_SIZE = 1;
    static constexpr int NDOFS = N_FIELD_COMPONENTS * N_SHAPE;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d();

    int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[NDOFS * NDOFS];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
        static constexpr int nelems = VECTOR_SIZE;
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
        scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                block_coordinate_data[shape * SPATIAL_DIM + d][0] = scalar_t(points[d][node]);
            }
        }

        scalar_t coordinate_grad_ref[SPATIAL_DIM * N_QP * SPATIAL_DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * SPATIAL_DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * SPATIAL_DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, SPATIAL_DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        laplace_d3_tensor_product_direct_hessian_tensor_product_element_matrix<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, kappa, element_matrix);

        if constexpr (FORMAT == 1) {
            laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
        } else if constexpr (FORMAT == 0) {
            laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
        } else if constexpr (FORMAT == 2) {
            laplace_proteus_hex729_hessian_isoparametric_mesh_soa_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values);
        } else {
            unsupported_matrix_format |= 1;
        }
    }

    return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 2>(nelements, nnodes, elements, points, kappa, nullptr, nullptr, values, diag_offsets, ndiag, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_proteus_hex729_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 2>(nelements, nnodes, elements, points, kappa, nullptr, nullptr, values, diag_offsets, ndiag, 0, nullptr, nullptr, nullptr, nullptr);
}
