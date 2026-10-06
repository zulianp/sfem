#ifndef LINEAR_ELASTICITY_PROTEUS_HEX729_ELEMENT_API_HPP
#define LINEAR_ELASTICITY_PROTEUS_HEX729_ELEMENT_API_HPP

#include <stddef.h>
#include "../linear_elasticity_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"

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


template <typename scalar_t>
struct linear_elasticity_proteus_hex729_isoparametric_reference_data {
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

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
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
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_energy_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
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
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_gradient_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
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
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex729_hessian_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 729;
    static constexpr int N_QP = 1000;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
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
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex729_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
