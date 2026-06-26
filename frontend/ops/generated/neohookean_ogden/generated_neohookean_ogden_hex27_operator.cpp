#include "generated_neohookean_ogden_d3_tensor_product_local.hpp"
#include "kernel_diagnostics.hpp"

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

static const real_t generated_neohookean_ogden_hex27_hex27_shape_1d[9] = {real_t(0.68729833462074175), real_t(0.39999999999999997), real_t(-0.087298334620741685), real_t(0), real_t(1), real_t(0), real_t(-0.087298334620741658), real_t(0.39999999999999991), real_t(0.68729833462074175)};
static const real_t generated_neohookean_ogden_hex27_hex27_grad_1d[9] = {real_t(-2.5491933384829668), real_t(3.0983866769659336), real_t(-0.54919333848296681), real_t(-1), real_t(0), real_t(1), real_t(0.54919333848296681), real_t(-3.0983866769659336), real_t(2.5491933384829668)};
static const real_t generated_neohookean_ogden_hex27_hex27_q_weight_1d[3] = {real_t(0.27777777777777779), real_t(0.44444444444444442), real_t(0.27777777777777779)};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data = {
    "generated_neohookean_ogden_hex27_hex27_objective_soa",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    2,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_hex27_hex27_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_objective_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};

        generated_neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_adjugate4,
        const real_t *const SFEM_RESTRICT jacobian_adjugate5,
        const real_t *const SFEM_RESTRICT jacobian_adjugate6,
        const real_t *const SFEM_RESTRICT jacobian_adjugate7,
        const real_t *const SFEM_RESTRICT jacobian_adjugate8,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_soa_impl<real_t, 27, 27, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_z0[lane] = z0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_z1[lane] = z1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_z2[lane] = z2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_z3[lane] = z3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_z4[lane] = z4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_z5[lane] = z5[evbegin + lane];
            block_x6[lane] = x6[evbegin + lane];
            block_y6[lane] = y6[evbegin + lane];
            block_z6[lane] = z6[evbegin + lane];
            block_x7[lane] = x7[evbegin + lane];
            block_y7[lane] = y7[evbegin + lane];
            block_z7[lane] = z7[evbegin + lane];
            block_x8[lane] = x8[evbegin + lane];
            block_y8[lane] = y8[evbegin + lane];
            block_z8[lane] = z8[evbegin + lane];
            block_x9[lane] = x9[evbegin + lane];
            block_y9[lane] = y9[evbegin + lane];
            block_z9[lane] = z9[evbegin + lane];
            block_x10[lane] = x10[evbegin + lane];
            block_y10[lane] = y10[evbegin + lane];
            block_z10[lane] = z10[evbegin + lane];
            block_x11[lane] = x11[evbegin + lane];
            block_y11[lane] = y11[evbegin + lane];
            block_z11[lane] = z11[evbegin + lane];
            block_x12[lane] = x12[evbegin + lane];
            block_y12[lane] = y12[evbegin + lane];
            block_z12[lane] = z12[evbegin + lane];
            block_x13[lane] = x13[evbegin + lane];
            block_y13[lane] = y13[evbegin + lane];
            block_z13[lane] = z13[evbegin + lane];
            block_x14[lane] = x14[evbegin + lane];
            block_y14[lane] = y14[evbegin + lane];
            block_z14[lane] = z14[evbegin + lane];
            block_x15[lane] = x15[evbegin + lane];
            block_y15[lane] = y15[evbegin + lane];
            block_z15[lane] = z15[evbegin + lane];
            block_x16[lane] = x16[evbegin + lane];
            block_y16[lane] = y16[evbegin + lane];
            block_z16[lane] = z16[evbegin + lane];
            block_x17[lane] = x17[evbegin + lane];
            block_y17[lane] = y17[evbegin + lane];
            block_z17[lane] = z17[evbegin + lane];
            block_x18[lane] = x18[evbegin + lane];
            block_y18[lane] = y18[evbegin + lane];
            block_z18[lane] = z18[evbegin + lane];
            block_x19[lane] = x19[evbegin + lane];
            block_y19[lane] = y19[evbegin + lane];
            block_z19[lane] = z19[evbegin + lane];
            block_x20[lane] = x20[evbegin + lane];
            block_y20[lane] = y20[evbegin + lane];
            block_z20[lane] = z20[evbegin + lane];
            block_x21[lane] = x21[evbegin + lane];
            block_y21[lane] = y21[evbegin + lane];
            block_z21[lane] = z21[evbegin + lane];
            block_x22[lane] = x22[evbegin + lane];
            block_y22[lane] = y22[evbegin + lane];
            block_z22[lane] = z22[evbegin + lane];
            block_x23[lane] = x23[evbegin + lane];
            block_y23[lane] = y23[evbegin + lane];
            block_z23[lane] = z23[evbegin + lane];
            block_x24[lane] = x24[evbegin + lane];
            block_y24[lane] = y24[evbegin + lane];
            block_z24[lane] = z24[evbegin + lane];
            block_x25[lane] = x25[evbegin + lane];
            block_y25[lane] = y25[evbegin + lane];
            block_z25[lane] = z25[evbegin + lane];
            block_x26[lane] = x26[evbegin + lane];
            block_y26[lane] = y26[evbegin + lane];
            block_z26[lane] = z26[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_isoparametric_soa_impl<real_t, 27, 27, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, x10, y10, z10, x11, y11, z11, x12, y12, z12, x13, y13, z13, x14, y14, z14, x15, y15, z15, x16, y16, z16, x17, y17, z17, x18, y18, z18, x19, y19, z19, x20, y20, z20, x21, y21, z21, x22, y22, z22, x23, y23, z23, x24, y24, z24, x25, y25, z25, x26, y26, z26, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};

        generated_neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_z0[lane] = z[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_z1[lane] = z[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_z2[lane] = z[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_z3[lane] = z[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_z4[lane] = z[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
            block_z5[lane] = z[ev[lane * N_SHAPE + 5]];
            block_x6[lane] = x[ev[lane * N_SHAPE + 6]];
            block_y6[lane] = y[ev[lane * N_SHAPE + 6]];
            block_z6[lane] = z[ev[lane * N_SHAPE + 6]];
            block_x7[lane] = x[ev[lane * N_SHAPE + 7]];
            block_y7[lane] = y[ev[lane * N_SHAPE + 7]];
            block_z7[lane] = z[ev[lane * N_SHAPE + 7]];
            block_x8[lane] = x[ev[lane * N_SHAPE + 8]];
            block_y8[lane] = y[ev[lane * N_SHAPE + 8]];
            block_z8[lane] = z[ev[lane * N_SHAPE + 8]];
            block_x9[lane] = x[ev[lane * N_SHAPE + 9]];
            block_y9[lane] = y[ev[lane * N_SHAPE + 9]];
            block_z9[lane] = z[ev[lane * N_SHAPE + 9]];
            block_x10[lane] = x[ev[lane * N_SHAPE + 10]];
            block_y10[lane] = y[ev[lane * N_SHAPE + 10]];
            block_z10[lane] = z[ev[lane * N_SHAPE + 10]];
            block_x11[lane] = x[ev[lane * N_SHAPE + 11]];
            block_y11[lane] = y[ev[lane * N_SHAPE + 11]];
            block_z11[lane] = z[ev[lane * N_SHAPE + 11]];
            block_x12[lane] = x[ev[lane * N_SHAPE + 12]];
            block_y12[lane] = y[ev[lane * N_SHAPE + 12]];
            block_z12[lane] = z[ev[lane * N_SHAPE + 12]];
            block_x13[lane] = x[ev[lane * N_SHAPE + 13]];
            block_y13[lane] = y[ev[lane * N_SHAPE + 13]];
            block_z13[lane] = z[ev[lane * N_SHAPE + 13]];
            block_x14[lane] = x[ev[lane * N_SHAPE + 14]];
            block_y14[lane] = y[ev[lane * N_SHAPE + 14]];
            block_z14[lane] = z[ev[lane * N_SHAPE + 14]];
            block_x15[lane] = x[ev[lane * N_SHAPE + 15]];
            block_y15[lane] = y[ev[lane * N_SHAPE + 15]];
            block_z15[lane] = z[ev[lane * N_SHAPE + 15]];
            block_x16[lane] = x[ev[lane * N_SHAPE + 16]];
            block_y16[lane] = y[ev[lane * N_SHAPE + 16]];
            block_z16[lane] = z[ev[lane * N_SHAPE + 16]];
            block_x17[lane] = x[ev[lane * N_SHAPE + 17]];
            block_y17[lane] = y[ev[lane * N_SHAPE + 17]];
            block_z17[lane] = z[ev[lane * N_SHAPE + 17]];
            block_x18[lane] = x[ev[lane * N_SHAPE + 18]];
            block_y18[lane] = y[ev[lane * N_SHAPE + 18]];
            block_z18[lane] = z[ev[lane * N_SHAPE + 18]];
            block_x19[lane] = x[ev[lane * N_SHAPE + 19]];
            block_y19[lane] = y[ev[lane * N_SHAPE + 19]];
            block_z19[lane] = z[ev[lane * N_SHAPE + 19]];
            block_x20[lane] = x[ev[lane * N_SHAPE + 20]];
            block_y20[lane] = y[ev[lane * N_SHAPE + 20]];
            block_z20[lane] = z[ev[lane * N_SHAPE + 20]];
            block_x21[lane] = x[ev[lane * N_SHAPE + 21]];
            block_y21[lane] = y[ev[lane * N_SHAPE + 21]];
            block_z21[lane] = z[ev[lane * N_SHAPE + 21]];
            block_x22[lane] = x[ev[lane * N_SHAPE + 22]];
            block_y22[lane] = y[ev[lane * N_SHAPE + 22]];
            block_z22[lane] = z[ev[lane * N_SHAPE + 22]];
            block_x23[lane] = x[ev[lane * N_SHAPE + 23]];
            block_y23[lane] = y[ev[lane * N_SHAPE + 23]];
            block_z23[lane] = z[ev[lane * N_SHAPE + 23]];
            block_x24[lane] = x[ev[lane * N_SHAPE + 24]];
            block_y24[lane] = y[ev[lane * N_SHAPE + 24]];
            block_z24[lane] = z[ev[lane * N_SHAPE + 24]];
            block_x25[lane] = x[ev[lane * N_SHAPE + 25]];
            block_y25[lane] = y[ev[lane * N_SHAPE + 25]];
            block_z25[lane] = z[ev[lane * N_SHAPE + 25]];
            block_x26[lane] = x[ev[lane * N_SHAPE + 26]];
            block_y26[lane] = y[ev[lane * N_SHAPE + 26]];
            block_z26[lane] = z[ev[lane * N_SHAPE + 26]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data = {
    "generated_neohookean_ogden_hex27_hex27_gradient_soa",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    2,
    81,
    0,
    81,
    81,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_hex27_hex27_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_gradient_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outz0[lane] = outz0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outz1[lane] = outz1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outz2[lane] = outz2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outz3[lane] = outz3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outz4[lane] = outz4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
            block_outz5[lane] = outz5[evbegin + lane];
            block_outx6[lane] = outx6[evbegin + lane];
            block_outy6[lane] = outy6[evbegin + lane];
            block_outz6[lane] = outz6[evbegin + lane];
            block_outx7[lane] = outx7[evbegin + lane];
            block_outy7[lane] = outy7[evbegin + lane];
            block_outz7[lane] = outz7[evbegin + lane];
            block_outx8[lane] = outx8[evbegin + lane];
            block_outy8[lane] = outy8[evbegin + lane];
            block_outz8[lane] = outz8[evbegin + lane];
            block_outx9[lane] = outx9[evbegin + lane];
            block_outy9[lane] = outy9[evbegin + lane];
            block_outz9[lane] = outz9[evbegin + lane];
            block_outx10[lane] = outx10[evbegin + lane];
            block_outy10[lane] = outy10[evbegin + lane];
            block_outz10[lane] = outz10[evbegin + lane];
            block_outx11[lane] = outx11[evbegin + lane];
            block_outy11[lane] = outy11[evbegin + lane];
            block_outz11[lane] = outz11[evbegin + lane];
            block_outx12[lane] = outx12[evbegin + lane];
            block_outy12[lane] = outy12[evbegin + lane];
            block_outz12[lane] = outz12[evbegin + lane];
            block_outx13[lane] = outx13[evbegin + lane];
            block_outy13[lane] = outy13[evbegin + lane];
            block_outz13[lane] = outz13[evbegin + lane];
            block_outx14[lane] = outx14[evbegin + lane];
            block_outy14[lane] = outy14[evbegin + lane];
            block_outz14[lane] = outz14[evbegin + lane];
            block_outx15[lane] = outx15[evbegin + lane];
            block_outy15[lane] = outy15[evbegin + lane];
            block_outz15[lane] = outz15[evbegin + lane];
            block_outx16[lane] = outx16[evbegin + lane];
            block_outy16[lane] = outy16[evbegin + lane];
            block_outz16[lane] = outz16[evbegin + lane];
            block_outx17[lane] = outx17[evbegin + lane];
            block_outy17[lane] = outy17[evbegin + lane];
            block_outz17[lane] = outz17[evbegin + lane];
            block_outx18[lane] = outx18[evbegin + lane];
            block_outy18[lane] = outy18[evbegin + lane];
            block_outz18[lane] = outz18[evbegin + lane];
            block_outx19[lane] = outx19[evbegin + lane];
            block_outy19[lane] = outy19[evbegin + lane];
            block_outz19[lane] = outz19[evbegin + lane];
            block_outx20[lane] = outx20[evbegin + lane];
            block_outy20[lane] = outy20[evbegin + lane];
            block_outz20[lane] = outz20[evbegin + lane];
            block_outx21[lane] = outx21[evbegin + lane];
            block_outy21[lane] = outy21[evbegin + lane];
            block_outz21[lane] = outz21[evbegin + lane];
            block_outx22[lane] = outx22[evbegin + lane];
            block_outy22[lane] = outy22[evbegin + lane];
            block_outz22[lane] = outz22[evbegin + lane];
            block_outx23[lane] = outx23[evbegin + lane];
            block_outy23[lane] = outy23[evbegin + lane];
            block_outz23[lane] = outz23[evbegin + lane];
            block_outx24[lane] = outx24[evbegin + lane];
            block_outy24[lane] = outy24[evbegin + lane];
            block_outz24[lane] = outz24[evbegin + lane];
            block_outx25[lane] = outx25[evbegin + lane];
            block_outy25[lane] = outy25[evbegin + lane];
            block_outz25[lane] = outz25[evbegin + lane];
            block_outx26[lane] = outx26[evbegin + lane];
            block_outy26[lane] = outy26[evbegin + lane];
            block_outz26[lane] = outz26[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        generated_neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outz0[evbegin + lane] = block_outz0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outz1[evbegin + lane] = block_outz1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outz2[evbegin + lane] = block_outz2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outz3[evbegin + lane] = block_outz3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outz4[evbegin + lane] = block_outz4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
            outz5[evbegin + lane] = block_outz5[lane];
            outx6[evbegin + lane] = block_outx6[lane];
            outy6[evbegin + lane] = block_outy6[lane];
            outz6[evbegin + lane] = block_outz6[lane];
            outx7[evbegin + lane] = block_outx7[lane];
            outy7[evbegin + lane] = block_outy7[lane];
            outz7[evbegin + lane] = block_outz7[lane];
            outx8[evbegin + lane] = block_outx8[lane];
            outy8[evbegin + lane] = block_outy8[lane];
            outz8[evbegin + lane] = block_outz8[lane];
            outx9[evbegin + lane] = block_outx9[lane];
            outy9[evbegin + lane] = block_outy9[lane];
            outz9[evbegin + lane] = block_outz9[lane];
            outx10[evbegin + lane] = block_outx10[lane];
            outy10[evbegin + lane] = block_outy10[lane];
            outz10[evbegin + lane] = block_outz10[lane];
            outx11[evbegin + lane] = block_outx11[lane];
            outy11[evbegin + lane] = block_outy11[lane];
            outz11[evbegin + lane] = block_outz11[lane];
            outx12[evbegin + lane] = block_outx12[lane];
            outy12[evbegin + lane] = block_outy12[lane];
            outz12[evbegin + lane] = block_outz12[lane];
            outx13[evbegin + lane] = block_outx13[lane];
            outy13[evbegin + lane] = block_outy13[lane];
            outz13[evbegin + lane] = block_outz13[lane];
            outx14[evbegin + lane] = block_outx14[lane];
            outy14[evbegin + lane] = block_outy14[lane];
            outz14[evbegin + lane] = block_outz14[lane];
            outx15[evbegin + lane] = block_outx15[lane];
            outy15[evbegin + lane] = block_outy15[lane];
            outz15[evbegin + lane] = block_outz15[lane];
            outx16[evbegin + lane] = block_outx16[lane];
            outy16[evbegin + lane] = block_outy16[lane];
            outz16[evbegin + lane] = block_outz16[lane];
            outx17[evbegin + lane] = block_outx17[lane];
            outy17[evbegin + lane] = block_outy17[lane];
            outz17[evbegin + lane] = block_outz17[lane];
            outx18[evbegin + lane] = block_outx18[lane];
            outy18[evbegin + lane] = block_outy18[lane];
            outz18[evbegin + lane] = block_outz18[lane];
            outx19[evbegin + lane] = block_outx19[lane];
            outy19[evbegin + lane] = block_outy19[lane];
            outz19[evbegin + lane] = block_outz19[lane];
            outx20[evbegin + lane] = block_outx20[lane];
            outy20[evbegin + lane] = block_outy20[lane];
            outz20[evbegin + lane] = block_outz20[lane];
            outx21[evbegin + lane] = block_outx21[lane];
            outy21[evbegin + lane] = block_outy21[lane];
            outz21[evbegin + lane] = block_outz21[lane];
            outx22[evbegin + lane] = block_outx22[lane];
            outy22[evbegin + lane] = block_outy22[lane];
            outz22[evbegin + lane] = block_outz22[lane];
            outx23[evbegin + lane] = block_outx23[lane];
            outy23[evbegin + lane] = block_outy23[lane];
            outz23[evbegin + lane] = block_outz23[lane];
            outx24[evbegin + lane] = block_outx24[lane];
            outy24[evbegin + lane] = block_outy24[lane];
            outz24[evbegin + lane] = block_outz24[lane];
            outx25[evbegin + lane] = block_outx25[lane];
            outy25[evbegin + lane] = block_outy25[lane];
            outz25[evbegin + lane] = block_outz25[lane];
            outx26[evbegin + lane] = block_outx26[lane];
            outy26[evbegin + lane] = block_outy26[lane];
            outz26[evbegin + lane] = block_outz26[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_adjugate4,
        const real_t *const SFEM_RESTRICT jacobian_adjugate5,
        const real_t *const SFEM_RESTRICT jacobian_adjugate6,
        const real_t *const SFEM_RESTRICT jacobian_adjugate7,
        const real_t *const SFEM_RESTRICT jacobian_adjugate8,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_soa_impl<real_t, 27, 27, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9, outx10, outy10, outz10, outx11, outy11, outz11, outx12, outy12, outz12, outx13, outy13, outz13, outx14, outy14, outz14, outx15, outy15, outz15, outx16, outy16, outz16, outx17, outy17, outz17, outx18, outy18, outz18, outx19, outy19, outz19, outx20, outy20, outz20, outx21, outy21, outz21, outx22, outy22, outz22, outx23, outy23, outz23, outx24, outy24, outz24, outx25, outy25, outz25, outx26, outy26, outz26);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_z0[lane] = z0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_z1[lane] = z1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_z2[lane] = z2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_z3[lane] = z3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_z4[lane] = z4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_z5[lane] = z5[evbegin + lane];
            block_x6[lane] = x6[evbegin + lane];
            block_y6[lane] = y6[evbegin + lane];
            block_z6[lane] = z6[evbegin + lane];
            block_x7[lane] = x7[evbegin + lane];
            block_y7[lane] = y7[evbegin + lane];
            block_z7[lane] = z7[evbegin + lane];
            block_x8[lane] = x8[evbegin + lane];
            block_y8[lane] = y8[evbegin + lane];
            block_z8[lane] = z8[evbegin + lane];
            block_x9[lane] = x9[evbegin + lane];
            block_y9[lane] = y9[evbegin + lane];
            block_z9[lane] = z9[evbegin + lane];
            block_x10[lane] = x10[evbegin + lane];
            block_y10[lane] = y10[evbegin + lane];
            block_z10[lane] = z10[evbegin + lane];
            block_x11[lane] = x11[evbegin + lane];
            block_y11[lane] = y11[evbegin + lane];
            block_z11[lane] = z11[evbegin + lane];
            block_x12[lane] = x12[evbegin + lane];
            block_y12[lane] = y12[evbegin + lane];
            block_z12[lane] = z12[evbegin + lane];
            block_x13[lane] = x13[evbegin + lane];
            block_y13[lane] = y13[evbegin + lane];
            block_z13[lane] = z13[evbegin + lane];
            block_x14[lane] = x14[evbegin + lane];
            block_y14[lane] = y14[evbegin + lane];
            block_z14[lane] = z14[evbegin + lane];
            block_x15[lane] = x15[evbegin + lane];
            block_y15[lane] = y15[evbegin + lane];
            block_z15[lane] = z15[evbegin + lane];
            block_x16[lane] = x16[evbegin + lane];
            block_y16[lane] = y16[evbegin + lane];
            block_z16[lane] = z16[evbegin + lane];
            block_x17[lane] = x17[evbegin + lane];
            block_y17[lane] = y17[evbegin + lane];
            block_z17[lane] = z17[evbegin + lane];
            block_x18[lane] = x18[evbegin + lane];
            block_y18[lane] = y18[evbegin + lane];
            block_z18[lane] = z18[evbegin + lane];
            block_x19[lane] = x19[evbegin + lane];
            block_y19[lane] = y19[evbegin + lane];
            block_z19[lane] = z19[evbegin + lane];
            block_x20[lane] = x20[evbegin + lane];
            block_y20[lane] = y20[evbegin + lane];
            block_z20[lane] = z20[evbegin + lane];
            block_x21[lane] = x21[evbegin + lane];
            block_y21[lane] = y21[evbegin + lane];
            block_z21[lane] = z21[evbegin + lane];
            block_x22[lane] = x22[evbegin + lane];
            block_y22[lane] = y22[evbegin + lane];
            block_z22[lane] = z22[evbegin + lane];
            block_x23[lane] = x23[evbegin + lane];
            block_y23[lane] = y23[evbegin + lane];
            block_z23[lane] = z23[evbegin + lane];
            block_x24[lane] = x24[evbegin + lane];
            block_y24[lane] = y24[evbegin + lane];
            block_z24[lane] = z24[evbegin + lane];
            block_x25[lane] = x25[evbegin + lane];
            block_y25[lane] = y25[evbegin + lane];
            block_z25[lane] = z25[evbegin + lane];
            block_x26[lane] = x26[evbegin + lane];
            block_y26[lane] = y26[evbegin + lane];
            block_z26[lane] = z26[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outz0[lane] = outz0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outz1[lane] = outz1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outz2[lane] = outz2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outz3[lane] = outz3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outz4[lane] = outz4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
            block_outz5[lane] = outz5[evbegin + lane];
            block_outx6[lane] = outx6[evbegin + lane];
            block_outy6[lane] = outy6[evbegin + lane];
            block_outz6[lane] = outz6[evbegin + lane];
            block_outx7[lane] = outx7[evbegin + lane];
            block_outy7[lane] = outy7[evbegin + lane];
            block_outz7[lane] = outz7[evbegin + lane];
            block_outx8[lane] = outx8[evbegin + lane];
            block_outy8[lane] = outy8[evbegin + lane];
            block_outz8[lane] = outz8[evbegin + lane];
            block_outx9[lane] = outx9[evbegin + lane];
            block_outy9[lane] = outy9[evbegin + lane];
            block_outz9[lane] = outz9[evbegin + lane];
            block_outx10[lane] = outx10[evbegin + lane];
            block_outy10[lane] = outy10[evbegin + lane];
            block_outz10[lane] = outz10[evbegin + lane];
            block_outx11[lane] = outx11[evbegin + lane];
            block_outy11[lane] = outy11[evbegin + lane];
            block_outz11[lane] = outz11[evbegin + lane];
            block_outx12[lane] = outx12[evbegin + lane];
            block_outy12[lane] = outy12[evbegin + lane];
            block_outz12[lane] = outz12[evbegin + lane];
            block_outx13[lane] = outx13[evbegin + lane];
            block_outy13[lane] = outy13[evbegin + lane];
            block_outz13[lane] = outz13[evbegin + lane];
            block_outx14[lane] = outx14[evbegin + lane];
            block_outy14[lane] = outy14[evbegin + lane];
            block_outz14[lane] = outz14[evbegin + lane];
            block_outx15[lane] = outx15[evbegin + lane];
            block_outy15[lane] = outy15[evbegin + lane];
            block_outz15[lane] = outz15[evbegin + lane];
            block_outx16[lane] = outx16[evbegin + lane];
            block_outy16[lane] = outy16[evbegin + lane];
            block_outz16[lane] = outz16[evbegin + lane];
            block_outx17[lane] = outx17[evbegin + lane];
            block_outy17[lane] = outy17[evbegin + lane];
            block_outz17[lane] = outz17[evbegin + lane];
            block_outx18[lane] = outx18[evbegin + lane];
            block_outy18[lane] = outy18[evbegin + lane];
            block_outz18[lane] = outz18[evbegin + lane];
            block_outx19[lane] = outx19[evbegin + lane];
            block_outy19[lane] = outy19[evbegin + lane];
            block_outz19[lane] = outz19[evbegin + lane];
            block_outx20[lane] = outx20[evbegin + lane];
            block_outy20[lane] = outy20[evbegin + lane];
            block_outz20[lane] = outz20[evbegin + lane];
            block_outx21[lane] = outx21[evbegin + lane];
            block_outy21[lane] = outy21[evbegin + lane];
            block_outz21[lane] = outz21[evbegin + lane];
            block_outx22[lane] = outx22[evbegin + lane];
            block_outy22[lane] = outy22[evbegin + lane];
            block_outz22[lane] = outz22[evbegin + lane];
            block_outx23[lane] = outx23[evbegin + lane];
            block_outy23[lane] = outy23[evbegin + lane];
            block_outz23[lane] = outz23[evbegin + lane];
            block_outx24[lane] = outx24[evbegin + lane];
            block_outy24[lane] = outy24[evbegin + lane];
            block_outz24[lane] = outz24[evbegin + lane];
            block_outx25[lane] = outx25[evbegin + lane];
            block_outy25[lane] = outy25[evbegin + lane];
            block_outz25[lane] = outz25[evbegin + lane];
            block_outx26[lane] = outx26[evbegin + lane];
            block_outy26[lane] = outy26[evbegin + lane];
            block_outz26[lane] = outz26[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outz0[evbegin + lane] = block_outz0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outz1[evbegin + lane] = block_outz1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outz2[evbegin + lane] = block_outz2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outz3[evbegin + lane] = block_outz3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outz4[evbegin + lane] = block_outz4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
            outz5[evbegin + lane] = block_outz5[lane];
            outx6[evbegin + lane] = block_outx6[lane];
            outy6[evbegin + lane] = block_outy6[lane];
            outz6[evbegin + lane] = block_outz6[lane];
            outx7[evbegin + lane] = block_outx7[lane];
            outy7[evbegin + lane] = block_outy7[lane];
            outz7[evbegin + lane] = block_outz7[lane];
            outx8[evbegin + lane] = block_outx8[lane];
            outy8[evbegin + lane] = block_outy8[lane];
            outz8[evbegin + lane] = block_outz8[lane];
            outx9[evbegin + lane] = block_outx9[lane];
            outy9[evbegin + lane] = block_outy9[lane];
            outz9[evbegin + lane] = block_outz9[lane];
            outx10[evbegin + lane] = block_outx10[lane];
            outy10[evbegin + lane] = block_outy10[lane];
            outz10[evbegin + lane] = block_outz10[lane];
            outx11[evbegin + lane] = block_outx11[lane];
            outy11[evbegin + lane] = block_outy11[lane];
            outz11[evbegin + lane] = block_outz11[lane];
            outx12[evbegin + lane] = block_outx12[lane];
            outy12[evbegin + lane] = block_outy12[lane];
            outz12[evbegin + lane] = block_outz12[lane];
            outx13[evbegin + lane] = block_outx13[lane];
            outy13[evbegin + lane] = block_outy13[lane];
            outz13[evbegin + lane] = block_outz13[lane];
            outx14[evbegin + lane] = block_outx14[lane];
            outy14[evbegin + lane] = block_outy14[lane];
            outz14[evbegin + lane] = block_outz14[lane];
            outx15[evbegin + lane] = block_outx15[lane];
            outy15[evbegin + lane] = block_outy15[lane];
            outz15[evbegin + lane] = block_outz15[lane];
            outx16[evbegin + lane] = block_outx16[lane];
            outy16[evbegin + lane] = block_outy16[lane];
            outz16[evbegin + lane] = block_outz16[lane];
            outx17[evbegin + lane] = block_outx17[lane];
            outy17[evbegin + lane] = block_outy17[lane];
            outz17[evbegin + lane] = block_outz17[lane];
            outx18[evbegin + lane] = block_outx18[lane];
            outy18[evbegin + lane] = block_outy18[lane];
            outz18[evbegin + lane] = block_outz18[lane];
            outx19[evbegin + lane] = block_outx19[lane];
            outy19[evbegin + lane] = block_outy19[lane];
            outz19[evbegin + lane] = block_outz19[lane];
            outx20[evbegin + lane] = block_outx20[lane];
            outy20[evbegin + lane] = block_outy20[lane];
            outz20[evbegin + lane] = block_outz20[lane];
            outx21[evbegin + lane] = block_outx21[lane];
            outy21[evbegin + lane] = block_outy21[lane];
            outz21[evbegin + lane] = block_outz21[lane];
            outx22[evbegin + lane] = block_outx22[lane];
            outy22[evbegin + lane] = block_outy22[lane];
            outz22[evbegin + lane] = block_outz22[lane];
            outx23[evbegin + lane] = block_outx23[lane];
            outy23[evbegin + lane] = block_outy23[lane];
            outz23[evbegin + lane] = block_outz23[lane];
            outx24[evbegin + lane] = block_outx24[lane];
            outy24[evbegin + lane] = block_outy24[lane];
            outz24[evbegin + lane] = block_outz24[lane];
            outx25[evbegin + lane] = block_outx25[lane];
            outy25[evbegin + lane] = block_outy25[lane];
            outz25[evbegin + lane] = block_outz25[lane];
            outx26[evbegin + lane] = block_outx26[lane];
            outy26[evbegin + lane] = block_outy26[lane];
            outz26[evbegin + lane] = block_outz26[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_soa_impl<real_t, 27, 27, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, x10, y10, z10, x11, y11, z11, x12, y12, z12, x13, y13, z13, x14, y14, z14, x15, y15, z15, x16, y16, z16, x17, y17, z17, x18, y18, z18, x19, y19, z19, x20, y20, z20, x21, y21, z21, x22, y22, z22, x23, y23, z23, x24, y24, z24, x25, y25, z25, x26, y26, z26, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9, outx10, outy10, outz10, outx11, outy11, outz11, outx12, outy12, outz12, outx13, outy13, outz13, outx14, outy14, outz14, outx15, outy15, outz15, outx16, outy16, outz16, outx17, outy17, outz17, outx18, outy18, outz18, outx19, outy19, outz19, outx20, outy20, outz20, outx21, outy21, outz21, outx22, outy22, outz22, outx23, outy23, outz23, outx24, outy24, outz24, outx25, outy25, outz25, outx26, outy26, outz26);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outz0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outz1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outz2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outz3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outz4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
            block_outz5[lane] = scalar_t(0);
            block_outx6[lane] = scalar_t(0);
            block_outy6[lane] = scalar_t(0);
            block_outz6[lane] = scalar_t(0);
            block_outx7[lane] = scalar_t(0);
            block_outy7[lane] = scalar_t(0);
            block_outz7[lane] = scalar_t(0);
            block_outx8[lane] = scalar_t(0);
            block_outy8[lane] = scalar_t(0);
            block_outz8[lane] = scalar_t(0);
            block_outx9[lane] = scalar_t(0);
            block_outy9[lane] = scalar_t(0);
            block_outz9[lane] = scalar_t(0);
            block_outx10[lane] = scalar_t(0);
            block_outy10[lane] = scalar_t(0);
            block_outz10[lane] = scalar_t(0);
            block_outx11[lane] = scalar_t(0);
            block_outy11[lane] = scalar_t(0);
            block_outz11[lane] = scalar_t(0);
            block_outx12[lane] = scalar_t(0);
            block_outy12[lane] = scalar_t(0);
            block_outz12[lane] = scalar_t(0);
            block_outx13[lane] = scalar_t(0);
            block_outy13[lane] = scalar_t(0);
            block_outz13[lane] = scalar_t(0);
            block_outx14[lane] = scalar_t(0);
            block_outy14[lane] = scalar_t(0);
            block_outz14[lane] = scalar_t(0);
            block_outx15[lane] = scalar_t(0);
            block_outy15[lane] = scalar_t(0);
            block_outz15[lane] = scalar_t(0);
            block_outx16[lane] = scalar_t(0);
            block_outy16[lane] = scalar_t(0);
            block_outz16[lane] = scalar_t(0);
            block_outx17[lane] = scalar_t(0);
            block_outy17[lane] = scalar_t(0);
            block_outz17[lane] = scalar_t(0);
            block_outx18[lane] = scalar_t(0);
            block_outy18[lane] = scalar_t(0);
            block_outz18[lane] = scalar_t(0);
            block_outx19[lane] = scalar_t(0);
            block_outy19[lane] = scalar_t(0);
            block_outz19[lane] = scalar_t(0);
            block_outx20[lane] = scalar_t(0);
            block_outy20[lane] = scalar_t(0);
            block_outz20[lane] = scalar_t(0);
            block_outx21[lane] = scalar_t(0);
            block_outy21[lane] = scalar_t(0);
            block_outz21[lane] = scalar_t(0);
            block_outx22[lane] = scalar_t(0);
            block_outy22[lane] = scalar_t(0);
            block_outz22[lane] = scalar_t(0);
            block_outx23[lane] = scalar_t(0);
            block_outy23[lane] = scalar_t(0);
            block_outz23[lane] = scalar_t(0);
            block_outx24[lane] = scalar_t(0);
            block_outy24[lane] = scalar_t(0);
            block_outz24[lane] = scalar_t(0);
            block_outx25[lane] = scalar_t(0);
            block_outy25[lane] = scalar_t(0);
            block_outz25[lane] = scalar_t(0);
            block_outx26[lane] = scalar_t(0);
            block_outy26[lane] = scalar_t(0);
            block_outz26[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        generated_neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 0] * out_stride] += block_outz0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 1] * out_stride] += block_outz1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 2] * out_stride] += block_outz2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 3] * out_stride] += block_outz3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 4] * out_stride] += block_outz4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 5] * out_stride] += block_outz5[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 6] * out_stride] += block_outx6[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 6] * out_stride] += block_outy6[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 6] * out_stride] += block_outz6[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 7] * out_stride] += block_outx7[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 7] * out_stride] += block_outy7[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 7] * out_stride] += block_outz7[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 8] * out_stride] += block_outx8[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 8] * out_stride] += block_outy8[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 8] * out_stride] += block_outz8[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 9] * out_stride] += block_outx9[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 9] * out_stride] += block_outy9[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 9] * out_stride] += block_outz9[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 10] * out_stride] += block_outx10[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 10] * out_stride] += block_outy10[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 10] * out_stride] += block_outz10[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 11] * out_stride] += block_outx11[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 11] * out_stride] += block_outy11[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 11] * out_stride] += block_outz11[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 12] * out_stride] += block_outx12[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 12] * out_stride] += block_outy12[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 12] * out_stride] += block_outz12[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 13] * out_stride] += block_outx13[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 13] * out_stride] += block_outy13[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 13] * out_stride] += block_outz13[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 14] * out_stride] += block_outx14[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 14] * out_stride] += block_outy14[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 14] * out_stride] += block_outz14[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 15] * out_stride] += block_outx15[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 15] * out_stride] += block_outy15[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 15] * out_stride] += block_outz15[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 16] * out_stride] += block_outx16[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 16] * out_stride] += block_outy16[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 16] * out_stride] += block_outz16[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 17] * out_stride] += block_outx17[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 17] * out_stride] += block_outy17[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 17] * out_stride] += block_outz17[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 18] * out_stride] += block_outx18[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 18] * out_stride] += block_outy18[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 18] * out_stride] += block_outz18[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 19] * out_stride] += block_outx19[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 19] * out_stride] += block_outy19[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 19] * out_stride] += block_outz19[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 20] * out_stride] += block_outx20[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 20] * out_stride] += block_outy20[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 20] * out_stride] += block_outz20[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 21] * out_stride] += block_outx21[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 21] * out_stride] += block_outy21[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 21] * out_stride] += block_outz21[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 22] * out_stride] += block_outx22[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 22] * out_stride] += block_outy22[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 22] * out_stride] += block_outz22[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 23] * out_stride] += block_outx23[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 23] * out_stride] += block_outy23[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 23] * out_stride] += block_outz23[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 24] * out_stride] += block_outx24[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 24] * out_stride] += block_outy24[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 24] * out_stride] += block_outz24[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 25] * out_stride] += block_outx25[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 25] * out_stride] += block_outy25[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 25] * out_stride] += block_outz25[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 26] * out_stride] += block_outx26[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 26] * out_stride] += block_outy26[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 26] * out_stride] += block_outz26[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_z0[lane] = z[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_z1[lane] = z[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_z2[lane] = z[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_z3[lane] = z[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_z4[lane] = z[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
            block_z5[lane] = z[ev[lane * N_SHAPE + 5]];
            block_x6[lane] = x[ev[lane * N_SHAPE + 6]];
            block_y6[lane] = y[ev[lane * N_SHAPE + 6]];
            block_z6[lane] = z[ev[lane * N_SHAPE + 6]];
            block_x7[lane] = x[ev[lane * N_SHAPE + 7]];
            block_y7[lane] = y[ev[lane * N_SHAPE + 7]];
            block_z7[lane] = z[ev[lane * N_SHAPE + 7]];
            block_x8[lane] = x[ev[lane * N_SHAPE + 8]];
            block_y8[lane] = y[ev[lane * N_SHAPE + 8]];
            block_z8[lane] = z[ev[lane * N_SHAPE + 8]];
            block_x9[lane] = x[ev[lane * N_SHAPE + 9]];
            block_y9[lane] = y[ev[lane * N_SHAPE + 9]];
            block_z9[lane] = z[ev[lane * N_SHAPE + 9]];
            block_x10[lane] = x[ev[lane * N_SHAPE + 10]];
            block_y10[lane] = y[ev[lane * N_SHAPE + 10]];
            block_z10[lane] = z[ev[lane * N_SHAPE + 10]];
            block_x11[lane] = x[ev[lane * N_SHAPE + 11]];
            block_y11[lane] = y[ev[lane * N_SHAPE + 11]];
            block_z11[lane] = z[ev[lane * N_SHAPE + 11]];
            block_x12[lane] = x[ev[lane * N_SHAPE + 12]];
            block_y12[lane] = y[ev[lane * N_SHAPE + 12]];
            block_z12[lane] = z[ev[lane * N_SHAPE + 12]];
            block_x13[lane] = x[ev[lane * N_SHAPE + 13]];
            block_y13[lane] = y[ev[lane * N_SHAPE + 13]];
            block_z13[lane] = z[ev[lane * N_SHAPE + 13]];
            block_x14[lane] = x[ev[lane * N_SHAPE + 14]];
            block_y14[lane] = y[ev[lane * N_SHAPE + 14]];
            block_z14[lane] = z[ev[lane * N_SHAPE + 14]];
            block_x15[lane] = x[ev[lane * N_SHAPE + 15]];
            block_y15[lane] = y[ev[lane * N_SHAPE + 15]];
            block_z15[lane] = z[ev[lane * N_SHAPE + 15]];
            block_x16[lane] = x[ev[lane * N_SHAPE + 16]];
            block_y16[lane] = y[ev[lane * N_SHAPE + 16]];
            block_z16[lane] = z[ev[lane * N_SHAPE + 16]];
            block_x17[lane] = x[ev[lane * N_SHAPE + 17]];
            block_y17[lane] = y[ev[lane * N_SHAPE + 17]];
            block_z17[lane] = z[ev[lane * N_SHAPE + 17]];
            block_x18[lane] = x[ev[lane * N_SHAPE + 18]];
            block_y18[lane] = y[ev[lane * N_SHAPE + 18]];
            block_z18[lane] = z[ev[lane * N_SHAPE + 18]];
            block_x19[lane] = x[ev[lane * N_SHAPE + 19]];
            block_y19[lane] = y[ev[lane * N_SHAPE + 19]];
            block_z19[lane] = z[ev[lane * N_SHAPE + 19]];
            block_x20[lane] = x[ev[lane * N_SHAPE + 20]];
            block_y20[lane] = y[ev[lane * N_SHAPE + 20]];
            block_z20[lane] = z[ev[lane * N_SHAPE + 20]];
            block_x21[lane] = x[ev[lane * N_SHAPE + 21]];
            block_y21[lane] = y[ev[lane * N_SHAPE + 21]];
            block_z21[lane] = z[ev[lane * N_SHAPE + 21]];
            block_x22[lane] = x[ev[lane * N_SHAPE + 22]];
            block_y22[lane] = y[ev[lane * N_SHAPE + 22]];
            block_z22[lane] = z[ev[lane * N_SHAPE + 22]];
            block_x23[lane] = x[ev[lane * N_SHAPE + 23]];
            block_y23[lane] = y[ev[lane * N_SHAPE + 23]];
            block_z23[lane] = z[ev[lane * N_SHAPE + 23]];
            block_x24[lane] = x[ev[lane * N_SHAPE + 24]];
            block_y24[lane] = y[ev[lane * N_SHAPE + 24]];
            block_z24[lane] = z[ev[lane * N_SHAPE + 24]];
            block_x25[lane] = x[ev[lane * N_SHAPE + 25]];
            block_y25[lane] = y[ev[lane * N_SHAPE + 25]];
            block_z25[lane] = z[ev[lane * N_SHAPE + 25]];
            block_x26[lane] = x[ev[lane * N_SHAPE + 26]];
            block_y26[lane] = y[ev[lane * N_SHAPE + 26]];
            block_z26[lane] = z[ev[lane * N_SHAPE + 26]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outz0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outz1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outz2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outz3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outz4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
            block_outz5[lane] = scalar_t(0);
            block_outx6[lane] = scalar_t(0);
            block_outy6[lane] = scalar_t(0);
            block_outz6[lane] = scalar_t(0);
            block_outx7[lane] = scalar_t(0);
            block_outy7[lane] = scalar_t(0);
            block_outz7[lane] = scalar_t(0);
            block_outx8[lane] = scalar_t(0);
            block_outy8[lane] = scalar_t(0);
            block_outz8[lane] = scalar_t(0);
            block_outx9[lane] = scalar_t(0);
            block_outy9[lane] = scalar_t(0);
            block_outz9[lane] = scalar_t(0);
            block_outx10[lane] = scalar_t(0);
            block_outy10[lane] = scalar_t(0);
            block_outz10[lane] = scalar_t(0);
            block_outx11[lane] = scalar_t(0);
            block_outy11[lane] = scalar_t(0);
            block_outz11[lane] = scalar_t(0);
            block_outx12[lane] = scalar_t(0);
            block_outy12[lane] = scalar_t(0);
            block_outz12[lane] = scalar_t(0);
            block_outx13[lane] = scalar_t(0);
            block_outy13[lane] = scalar_t(0);
            block_outz13[lane] = scalar_t(0);
            block_outx14[lane] = scalar_t(0);
            block_outy14[lane] = scalar_t(0);
            block_outz14[lane] = scalar_t(0);
            block_outx15[lane] = scalar_t(0);
            block_outy15[lane] = scalar_t(0);
            block_outz15[lane] = scalar_t(0);
            block_outx16[lane] = scalar_t(0);
            block_outy16[lane] = scalar_t(0);
            block_outz16[lane] = scalar_t(0);
            block_outx17[lane] = scalar_t(0);
            block_outy17[lane] = scalar_t(0);
            block_outz17[lane] = scalar_t(0);
            block_outx18[lane] = scalar_t(0);
            block_outy18[lane] = scalar_t(0);
            block_outz18[lane] = scalar_t(0);
            block_outx19[lane] = scalar_t(0);
            block_outy19[lane] = scalar_t(0);
            block_outz19[lane] = scalar_t(0);
            block_outx20[lane] = scalar_t(0);
            block_outy20[lane] = scalar_t(0);
            block_outz20[lane] = scalar_t(0);
            block_outx21[lane] = scalar_t(0);
            block_outy21[lane] = scalar_t(0);
            block_outz21[lane] = scalar_t(0);
            block_outx22[lane] = scalar_t(0);
            block_outy22[lane] = scalar_t(0);
            block_outz22[lane] = scalar_t(0);
            block_outx23[lane] = scalar_t(0);
            block_outy23[lane] = scalar_t(0);
            block_outz23[lane] = scalar_t(0);
            block_outx24[lane] = scalar_t(0);
            block_outy24[lane] = scalar_t(0);
            block_outz24[lane] = scalar_t(0);
            block_outx25[lane] = scalar_t(0);
            block_outy25[lane] = scalar_t(0);
            block_outz25[lane] = scalar_t(0);
            block_outx26[lane] = scalar_t(0);
            block_outy26[lane] = scalar_t(0);
            block_outz26[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 0] * out_stride] += block_outz0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 1] * out_stride] += block_outz1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 2] * out_stride] += block_outz2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 3] * out_stride] += block_outz3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 4] * out_stride] += block_outz4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 5] * out_stride] += block_outz5[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 6] * out_stride] += block_outx6[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 6] * out_stride] += block_outy6[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 6] * out_stride] += block_outz6[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 7] * out_stride] += block_outx7[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 7] * out_stride] += block_outy7[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 7] * out_stride] += block_outz7[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 8] * out_stride] += block_outx8[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 8] * out_stride] += block_outy8[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 8] * out_stride] += block_outz8[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 9] * out_stride] += block_outx9[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 9] * out_stride] += block_outy9[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 9] * out_stride] += block_outz9[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 10] * out_stride] += block_outx10[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 10] * out_stride] += block_outy10[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 10] * out_stride] += block_outz10[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 11] * out_stride] += block_outx11[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 11] * out_stride] += block_outy11[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 11] * out_stride] += block_outz11[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 12] * out_stride] += block_outx12[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 12] * out_stride] += block_outy12[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 12] * out_stride] += block_outz12[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 13] * out_stride] += block_outx13[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 13] * out_stride] += block_outy13[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 13] * out_stride] += block_outz13[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 14] * out_stride] += block_outx14[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 14] * out_stride] += block_outy14[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 14] * out_stride] += block_outz14[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 15] * out_stride] += block_outx15[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 15] * out_stride] += block_outy15[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 15] * out_stride] += block_outz15[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 16] * out_stride] += block_outx16[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 16] * out_stride] += block_outy16[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 16] * out_stride] += block_outz16[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 17] * out_stride] += block_outx17[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 17] * out_stride] += block_outy17[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 17] * out_stride] += block_outz17[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 18] * out_stride] += block_outx18[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 18] * out_stride] += block_outy18[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 18] * out_stride] += block_outz18[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 19] * out_stride] += block_outx19[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 19] * out_stride] += block_outy19[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 19] * out_stride] += block_outz19[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 20] * out_stride] += block_outx20[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 20] * out_stride] += block_outy20[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 20] * out_stride] += block_outz20[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 21] * out_stride] += block_outx21[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 21] * out_stride] += block_outy21[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 21] * out_stride] += block_outz21[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 22] * out_stride] += block_outx22[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 22] * out_stride] += block_outy22[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 22] * out_stride] += block_outz22[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 23] * out_stride] += block_outx23[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 23] * out_stride] += block_outy23[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 23] * out_stride] += block_outz23[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 24] * out_stride] += block_outx24[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 24] * out_stride] += block_outy24[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 24] * out_stride] += block_outz24[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 25] * out_stride] += block_outx25[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 25] * out_stride] += block_outy25[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 25] * out_stride] += block_outz25[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 26] * out_stride] += block_outx26[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 26] * out_stride] += block_outy26[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 26] * out_stride] += block_outz26[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data = {
    "generated_neohookean_ogden_hex27_hex27_apply_soa",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    2,
    81,
    81,
    81,
    81,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_hex27_hex27_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_apply_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hz0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hz1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hz2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hz3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hz4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        const real_t *const SFEM_RESTRICT hz5,
        const real_t *const SFEM_RESTRICT hx6,
        const real_t *const SFEM_RESTRICT hy6,
        const real_t *const SFEM_RESTRICT hz6,
        const real_t *const SFEM_RESTRICT hx7,
        const real_t *const SFEM_RESTRICT hy7,
        const real_t *const SFEM_RESTRICT hz7,
        const real_t *const SFEM_RESTRICT hx8,
        const real_t *const SFEM_RESTRICT hy8,
        const real_t *const SFEM_RESTRICT hz8,
        const real_t *const SFEM_RESTRICT hx9,
        const real_t *const SFEM_RESTRICT hy9,
        const real_t *const SFEM_RESTRICT hz9,
        const real_t *const SFEM_RESTRICT hx10,
        const real_t *const SFEM_RESTRICT hy10,
        const real_t *const SFEM_RESTRICT hz10,
        const real_t *const SFEM_RESTRICT hx11,
        const real_t *const SFEM_RESTRICT hy11,
        const real_t *const SFEM_RESTRICT hz11,
        const real_t *const SFEM_RESTRICT hx12,
        const real_t *const SFEM_RESTRICT hy12,
        const real_t *const SFEM_RESTRICT hz12,
        const real_t *const SFEM_RESTRICT hx13,
        const real_t *const SFEM_RESTRICT hy13,
        const real_t *const SFEM_RESTRICT hz13,
        const real_t *const SFEM_RESTRICT hx14,
        const real_t *const SFEM_RESTRICT hy14,
        const real_t *const SFEM_RESTRICT hz14,
        const real_t *const SFEM_RESTRICT hx15,
        const real_t *const SFEM_RESTRICT hy15,
        const real_t *const SFEM_RESTRICT hz15,
        const real_t *const SFEM_RESTRICT hx16,
        const real_t *const SFEM_RESTRICT hy16,
        const real_t *const SFEM_RESTRICT hz16,
        const real_t *const SFEM_RESTRICT hx17,
        const real_t *const SFEM_RESTRICT hy17,
        const real_t *const SFEM_RESTRICT hz17,
        const real_t *const SFEM_RESTRICT hx18,
        const real_t *const SFEM_RESTRICT hy18,
        const real_t *const SFEM_RESTRICT hz18,
        const real_t *const SFEM_RESTRICT hx19,
        const real_t *const SFEM_RESTRICT hy19,
        const real_t *const SFEM_RESTRICT hz19,
        const real_t *const SFEM_RESTRICT hx20,
        const real_t *const SFEM_RESTRICT hy20,
        const real_t *const SFEM_RESTRICT hz20,
        const real_t *const SFEM_RESTRICT hx21,
        const real_t *const SFEM_RESTRICT hy21,
        const real_t *const SFEM_RESTRICT hz21,
        const real_t *const SFEM_RESTRICT hx22,
        const real_t *const SFEM_RESTRICT hy22,
        const real_t *const SFEM_RESTRICT hz22,
        const real_t *const SFEM_RESTRICT hx23,
        const real_t *const SFEM_RESTRICT hy23,
        const real_t *const SFEM_RESTRICT hz23,
        const real_t *const SFEM_RESTRICT hx24,
        const real_t *const SFEM_RESTRICT hy24,
        const real_t *const SFEM_RESTRICT hz24,
        const real_t *const SFEM_RESTRICT hx25,
        const real_t *const SFEM_RESTRICT hy25,
        const real_t *const SFEM_RESTRICT hz25,
        const real_t *const SFEM_RESTRICT hx26,
        const real_t *const SFEM_RESTRICT hy26,
        const real_t *const SFEM_RESTRICT hz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hz0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hz1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hz2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hz3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hz4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_hz5[VECTOR_SIZE];
        scalar_t block_hx6[VECTOR_SIZE];
        scalar_t block_hy6[VECTOR_SIZE];
        scalar_t block_hz6[VECTOR_SIZE];
        scalar_t block_hx7[VECTOR_SIZE];
        scalar_t block_hy7[VECTOR_SIZE];
        scalar_t block_hz7[VECTOR_SIZE];
        scalar_t block_hx8[VECTOR_SIZE];
        scalar_t block_hy8[VECTOR_SIZE];
        scalar_t block_hz8[VECTOR_SIZE];
        scalar_t block_hx9[VECTOR_SIZE];
        scalar_t block_hy9[VECTOR_SIZE];
        scalar_t block_hz9[VECTOR_SIZE];
        scalar_t block_hx10[VECTOR_SIZE];
        scalar_t block_hy10[VECTOR_SIZE];
        scalar_t block_hz10[VECTOR_SIZE];
        scalar_t block_hx11[VECTOR_SIZE];
        scalar_t block_hy11[VECTOR_SIZE];
        scalar_t block_hz11[VECTOR_SIZE];
        scalar_t block_hx12[VECTOR_SIZE];
        scalar_t block_hy12[VECTOR_SIZE];
        scalar_t block_hz12[VECTOR_SIZE];
        scalar_t block_hx13[VECTOR_SIZE];
        scalar_t block_hy13[VECTOR_SIZE];
        scalar_t block_hz13[VECTOR_SIZE];
        scalar_t block_hx14[VECTOR_SIZE];
        scalar_t block_hy14[VECTOR_SIZE];
        scalar_t block_hz14[VECTOR_SIZE];
        scalar_t block_hx15[VECTOR_SIZE];
        scalar_t block_hy15[VECTOR_SIZE];
        scalar_t block_hz15[VECTOR_SIZE];
        scalar_t block_hx16[VECTOR_SIZE];
        scalar_t block_hy16[VECTOR_SIZE];
        scalar_t block_hz16[VECTOR_SIZE];
        scalar_t block_hx17[VECTOR_SIZE];
        scalar_t block_hy17[VECTOR_SIZE];
        scalar_t block_hz17[VECTOR_SIZE];
        scalar_t block_hx18[VECTOR_SIZE];
        scalar_t block_hy18[VECTOR_SIZE];
        scalar_t block_hz18[VECTOR_SIZE];
        scalar_t block_hx19[VECTOR_SIZE];
        scalar_t block_hy19[VECTOR_SIZE];
        scalar_t block_hz19[VECTOR_SIZE];
        scalar_t block_hx20[VECTOR_SIZE];
        scalar_t block_hy20[VECTOR_SIZE];
        scalar_t block_hz20[VECTOR_SIZE];
        scalar_t block_hx21[VECTOR_SIZE];
        scalar_t block_hy21[VECTOR_SIZE];
        scalar_t block_hz21[VECTOR_SIZE];
        scalar_t block_hx22[VECTOR_SIZE];
        scalar_t block_hy22[VECTOR_SIZE];
        scalar_t block_hz22[VECTOR_SIZE];
        scalar_t block_hx23[VECTOR_SIZE];
        scalar_t block_hy23[VECTOR_SIZE];
        scalar_t block_hz23[VECTOR_SIZE];
        scalar_t block_hx24[VECTOR_SIZE];
        scalar_t block_hy24[VECTOR_SIZE];
        scalar_t block_hz24[VECTOR_SIZE];
        scalar_t block_hx25[VECTOR_SIZE];
        scalar_t block_hy25[VECTOR_SIZE];
        scalar_t block_hz25[VECTOR_SIZE];
        scalar_t block_hx26[VECTOR_SIZE];
        scalar_t block_hy26[VECTOR_SIZE];
        scalar_t block_hz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_hx0[lane] = hx0[evbegin + lane];
            block_hy0[lane] = hy0[evbegin + lane];
            block_hz0[lane] = hz0[evbegin + lane];
            block_hx1[lane] = hx1[evbegin + lane];
            block_hy1[lane] = hy1[evbegin + lane];
            block_hz1[lane] = hz1[evbegin + lane];
            block_hx2[lane] = hx2[evbegin + lane];
            block_hy2[lane] = hy2[evbegin + lane];
            block_hz2[lane] = hz2[evbegin + lane];
            block_hx3[lane] = hx3[evbegin + lane];
            block_hy3[lane] = hy3[evbegin + lane];
            block_hz3[lane] = hz3[evbegin + lane];
            block_hx4[lane] = hx4[evbegin + lane];
            block_hy4[lane] = hy4[evbegin + lane];
            block_hz4[lane] = hz4[evbegin + lane];
            block_hx5[lane] = hx5[evbegin + lane];
            block_hy5[lane] = hy5[evbegin + lane];
            block_hz5[lane] = hz5[evbegin + lane];
            block_hx6[lane] = hx6[evbegin + lane];
            block_hy6[lane] = hy6[evbegin + lane];
            block_hz6[lane] = hz6[evbegin + lane];
            block_hx7[lane] = hx7[evbegin + lane];
            block_hy7[lane] = hy7[evbegin + lane];
            block_hz7[lane] = hz7[evbegin + lane];
            block_hx8[lane] = hx8[evbegin + lane];
            block_hy8[lane] = hy8[evbegin + lane];
            block_hz8[lane] = hz8[evbegin + lane];
            block_hx9[lane] = hx9[evbegin + lane];
            block_hy9[lane] = hy9[evbegin + lane];
            block_hz9[lane] = hz9[evbegin + lane];
            block_hx10[lane] = hx10[evbegin + lane];
            block_hy10[lane] = hy10[evbegin + lane];
            block_hz10[lane] = hz10[evbegin + lane];
            block_hx11[lane] = hx11[evbegin + lane];
            block_hy11[lane] = hy11[evbegin + lane];
            block_hz11[lane] = hz11[evbegin + lane];
            block_hx12[lane] = hx12[evbegin + lane];
            block_hy12[lane] = hy12[evbegin + lane];
            block_hz12[lane] = hz12[evbegin + lane];
            block_hx13[lane] = hx13[evbegin + lane];
            block_hy13[lane] = hy13[evbegin + lane];
            block_hz13[lane] = hz13[evbegin + lane];
            block_hx14[lane] = hx14[evbegin + lane];
            block_hy14[lane] = hy14[evbegin + lane];
            block_hz14[lane] = hz14[evbegin + lane];
            block_hx15[lane] = hx15[evbegin + lane];
            block_hy15[lane] = hy15[evbegin + lane];
            block_hz15[lane] = hz15[evbegin + lane];
            block_hx16[lane] = hx16[evbegin + lane];
            block_hy16[lane] = hy16[evbegin + lane];
            block_hz16[lane] = hz16[evbegin + lane];
            block_hx17[lane] = hx17[evbegin + lane];
            block_hy17[lane] = hy17[evbegin + lane];
            block_hz17[lane] = hz17[evbegin + lane];
            block_hx18[lane] = hx18[evbegin + lane];
            block_hy18[lane] = hy18[evbegin + lane];
            block_hz18[lane] = hz18[evbegin + lane];
            block_hx19[lane] = hx19[evbegin + lane];
            block_hy19[lane] = hy19[evbegin + lane];
            block_hz19[lane] = hz19[evbegin + lane];
            block_hx20[lane] = hx20[evbegin + lane];
            block_hy20[lane] = hy20[evbegin + lane];
            block_hz20[lane] = hz20[evbegin + lane];
            block_hx21[lane] = hx21[evbegin + lane];
            block_hy21[lane] = hy21[evbegin + lane];
            block_hz21[lane] = hz21[evbegin + lane];
            block_hx22[lane] = hx22[evbegin + lane];
            block_hy22[lane] = hy22[evbegin + lane];
            block_hz22[lane] = hz22[evbegin + lane];
            block_hx23[lane] = hx23[evbegin + lane];
            block_hy23[lane] = hy23[evbegin + lane];
            block_hz23[lane] = hz23[evbegin + lane];
            block_hx24[lane] = hx24[evbegin + lane];
            block_hy24[lane] = hy24[evbegin + lane];
            block_hz24[lane] = hz24[evbegin + lane];
            block_hx25[lane] = hx25[evbegin + lane];
            block_hy25[lane] = hy25[evbegin + lane];
            block_hz25[lane] = hz25[evbegin + lane];
            block_hx26[lane] = hx26[evbegin + lane];
            block_hy26[lane] = hy26[evbegin + lane];
            block_hz26[lane] = hz26[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outz0[lane] = outz0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outz1[lane] = outz1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outz2[lane] = outz2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outz3[lane] = outz3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outz4[lane] = outz4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
            block_outz5[lane] = outz5[evbegin + lane];
            block_outx6[lane] = outx6[evbegin + lane];
            block_outy6[lane] = outy6[evbegin + lane];
            block_outz6[lane] = outz6[evbegin + lane];
            block_outx7[lane] = outx7[evbegin + lane];
            block_outy7[lane] = outy7[evbegin + lane];
            block_outz7[lane] = outz7[evbegin + lane];
            block_outx8[lane] = outx8[evbegin + lane];
            block_outy8[lane] = outy8[evbegin + lane];
            block_outz8[lane] = outz8[evbegin + lane];
            block_outx9[lane] = outx9[evbegin + lane];
            block_outy9[lane] = outy9[evbegin + lane];
            block_outz9[lane] = outz9[evbegin + lane];
            block_outx10[lane] = outx10[evbegin + lane];
            block_outy10[lane] = outy10[evbegin + lane];
            block_outz10[lane] = outz10[evbegin + lane];
            block_outx11[lane] = outx11[evbegin + lane];
            block_outy11[lane] = outy11[evbegin + lane];
            block_outz11[lane] = outz11[evbegin + lane];
            block_outx12[lane] = outx12[evbegin + lane];
            block_outy12[lane] = outy12[evbegin + lane];
            block_outz12[lane] = outz12[evbegin + lane];
            block_outx13[lane] = outx13[evbegin + lane];
            block_outy13[lane] = outy13[evbegin + lane];
            block_outz13[lane] = outz13[evbegin + lane];
            block_outx14[lane] = outx14[evbegin + lane];
            block_outy14[lane] = outy14[evbegin + lane];
            block_outz14[lane] = outz14[evbegin + lane];
            block_outx15[lane] = outx15[evbegin + lane];
            block_outy15[lane] = outy15[evbegin + lane];
            block_outz15[lane] = outz15[evbegin + lane];
            block_outx16[lane] = outx16[evbegin + lane];
            block_outy16[lane] = outy16[evbegin + lane];
            block_outz16[lane] = outz16[evbegin + lane];
            block_outx17[lane] = outx17[evbegin + lane];
            block_outy17[lane] = outy17[evbegin + lane];
            block_outz17[lane] = outz17[evbegin + lane];
            block_outx18[lane] = outx18[evbegin + lane];
            block_outy18[lane] = outy18[evbegin + lane];
            block_outz18[lane] = outz18[evbegin + lane];
            block_outx19[lane] = outx19[evbegin + lane];
            block_outy19[lane] = outy19[evbegin + lane];
            block_outz19[lane] = outz19[evbegin + lane];
            block_outx20[lane] = outx20[evbegin + lane];
            block_outy20[lane] = outy20[evbegin + lane];
            block_outz20[lane] = outz20[evbegin + lane];
            block_outx21[lane] = outx21[evbegin + lane];
            block_outy21[lane] = outy21[evbegin + lane];
            block_outz21[lane] = outz21[evbegin + lane];
            block_outx22[lane] = outx22[evbegin + lane];
            block_outy22[lane] = outy22[evbegin + lane];
            block_outz22[lane] = outz22[evbegin + lane];
            block_outx23[lane] = outx23[evbegin + lane];
            block_outy23[lane] = outy23[evbegin + lane];
            block_outz23[lane] = outz23[evbegin + lane];
            block_outx24[lane] = outx24[evbegin + lane];
            block_outy24[lane] = outy24[evbegin + lane];
            block_outz24[lane] = outz24[evbegin + lane];
            block_outx25[lane] = outx25[evbegin + lane];
            block_outy25[lane] = outy25[evbegin + lane];
            block_outz25[lane] = outz25[evbegin + lane];
            block_outx26[lane] = outx26[evbegin + lane];
            block_outy26[lane] = outy26[evbegin + lane];
            block_outz26[lane] = outz26[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx8, block_hy8, block_hz8, block_hx1, block_hy1, block_hz1, block_hx11, block_hy11, block_hz11, block_hx24, block_hy24, block_hz24, block_hx9, block_hy9, block_hz9, block_hx3, block_hy3, block_hz3, block_hx10, block_hy10, block_hz10, block_hx2, block_hy2, block_hz2, block_hx16, block_hy16, block_hz16, block_hx20, block_hy20, block_hz20, block_hx17, block_hy17, block_hz17, block_hx23, block_hy23, block_hz23, block_hx26, block_hy26, block_hz26, block_hx21, block_hy21, block_hz21, block_hx19, block_hy19, block_hz19, block_hx22, block_hy22, block_hz22, block_hx18, block_hy18, block_hz18, block_hx4, block_hy4, block_hz4, block_hx12, block_hy12, block_hz12, block_hx5, block_hy5, block_hz5, block_hx15, block_hy15, block_hz15, block_hx25, block_hy25, block_hz25, block_hx13, block_hy13, block_hz13, block_hx7, block_hy7, block_hz7, block_hx14, block_hy14, block_hz14, block_hx6, block_hy6, block_hz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        generated_neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outz0[evbegin + lane] = block_outz0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outz1[evbegin + lane] = block_outz1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outz2[evbegin + lane] = block_outz2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outz3[evbegin + lane] = block_outz3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outz4[evbegin + lane] = block_outz4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
            outz5[evbegin + lane] = block_outz5[lane];
            outx6[evbegin + lane] = block_outx6[lane];
            outy6[evbegin + lane] = block_outy6[lane];
            outz6[evbegin + lane] = block_outz6[lane];
            outx7[evbegin + lane] = block_outx7[lane];
            outy7[evbegin + lane] = block_outy7[lane];
            outz7[evbegin + lane] = block_outz7[lane];
            outx8[evbegin + lane] = block_outx8[lane];
            outy8[evbegin + lane] = block_outy8[lane];
            outz8[evbegin + lane] = block_outz8[lane];
            outx9[evbegin + lane] = block_outx9[lane];
            outy9[evbegin + lane] = block_outy9[lane];
            outz9[evbegin + lane] = block_outz9[lane];
            outx10[evbegin + lane] = block_outx10[lane];
            outy10[evbegin + lane] = block_outy10[lane];
            outz10[evbegin + lane] = block_outz10[lane];
            outx11[evbegin + lane] = block_outx11[lane];
            outy11[evbegin + lane] = block_outy11[lane];
            outz11[evbegin + lane] = block_outz11[lane];
            outx12[evbegin + lane] = block_outx12[lane];
            outy12[evbegin + lane] = block_outy12[lane];
            outz12[evbegin + lane] = block_outz12[lane];
            outx13[evbegin + lane] = block_outx13[lane];
            outy13[evbegin + lane] = block_outy13[lane];
            outz13[evbegin + lane] = block_outz13[lane];
            outx14[evbegin + lane] = block_outx14[lane];
            outy14[evbegin + lane] = block_outy14[lane];
            outz14[evbegin + lane] = block_outz14[lane];
            outx15[evbegin + lane] = block_outx15[lane];
            outy15[evbegin + lane] = block_outy15[lane];
            outz15[evbegin + lane] = block_outz15[lane];
            outx16[evbegin + lane] = block_outx16[lane];
            outy16[evbegin + lane] = block_outy16[lane];
            outz16[evbegin + lane] = block_outz16[lane];
            outx17[evbegin + lane] = block_outx17[lane];
            outy17[evbegin + lane] = block_outy17[lane];
            outz17[evbegin + lane] = block_outz17[lane];
            outx18[evbegin + lane] = block_outx18[lane];
            outy18[evbegin + lane] = block_outy18[lane];
            outz18[evbegin + lane] = block_outz18[lane];
            outx19[evbegin + lane] = block_outx19[lane];
            outy19[evbegin + lane] = block_outy19[lane];
            outz19[evbegin + lane] = block_outz19[lane];
            outx20[evbegin + lane] = block_outx20[lane];
            outy20[evbegin + lane] = block_outy20[lane];
            outz20[evbegin + lane] = block_outz20[lane];
            outx21[evbegin + lane] = block_outx21[lane];
            outy21[evbegin + lane] = block_outy21[lane];
            outz21[evbegin + lane] = block_outz21[lane];
            outx22[evbegin + lane] = block_outx22[lane];
            outy22[evbegin + lane] = block_outy22[lane];
            outz22[evbegin + lane] = block_outz22[lane];
            outx23[evbegin + lane] = block_outx23[lane];
            outy23[evbegin + lane] = block_outy23[lane];
            outz23[evbegin + lane] = block_outz23[lane];
            outx24[evbegin + lane] = block_outx24[lane];
            outy24[evbegin + lane] = block_outy24[lane];
            outz24[evbegin + lane] = block_outz24[lane];
            outx25[evbegin + lane] = block_outx25[lane];
            outy25[evbegin + lane] = block_outy25[lane];
            outz25[evbegin + lane] = block_outz25[lane];
            outx26[evbegin + lane] = block_outx26[lane];
            outy26[evbegin + lane] = block_outy26[lane];
            outz26[evbegin + lane] = block_outz26[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_adjugate4,
        const real_t *const SFEM_RESTRICT jacobian_adjugate5,
        const real_t *const SFEM_RESTRICT jacobian_adjugate6,
        const real_t *const SFEM_RESTRICT jacobian_adjugate7,
        const real_t *const SFEM_RESTRICT jacobian_adjugate8,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hz0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hz1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hz2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hz3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hz4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        const real_t *const SFEM_RESTRICT hz5,
        const real_t *const SFEM_RESTRICT hx6,
        const real_t *const SFEM_RESTRICT hy6,
        const real_t *const SFEM_RESTRICT hz6,
        const real_t *const SFEM_RESTRICT hx7,
        const real_t *const SFEM_RESTRICT hy7,
        const real_t *const SFEM_RESTRICT hz7,
        const real_t *const SFEM_RESTRICT hx8,
        const real_t *const SFEM_RESTRICT hy8,
        const real_t *const SFEM_RESTRICT hz8,
        const real_t *const SFEM_RESTRICT hx9,
        const real_t *const SFEM_RESTRICT hy9,
        const real_t *const SFEM_RESTRICT hz9,
        const real_t *const SFEM_RESTRICT hx10,
        const real_t *const SFEM_RESTRICT hy10,
        const real_t *const SFEM_RESTRICT hz10,
        const real_t *const SFEM_RESTRICT hx11,
        const real_t *const SFEM_RESTRICT hy11,
        const real_t *const SFEM_RESTRICT hz11,
        const real_t *const SFEM_RESTRICT hx12,
        const real_t *const SFEM_RESTRICT hy12,
        const real_t *const SFEM_RESTRICT hz12,
        const real_t *const SFEM_RESTRICT hx13,
        const real_t *const SFEM_RESTRICT hy13,
        const real_t *const SFEM_RESTRICT hz13,
        const real_t *const SFEM_RESTRICT hx14,
        const real_t *const SFEM_RESTRICT hy14,
        const real_t *const SFEM_RESTRICT hz14,
        const real_t *const SFEM_RESTRICT hx15,
        const real_t *const SFEM_RESTRICT hy15,
        const real_t *const SFEM_RESTRICT hz15,
        const real_t *const SFEM_RESTRICT hx16,
        const real_t *const SFEM_RESTRICT hy16,
        const real_t *const SFEM_RESTRICT hz16,
        const real_t *const SFEM_RESTRICT hx17,
        const real_t *const SFEM_RESTRICT hy17,
        const real_t *const SFEM_RESTRICT hz17,
        const real_t *const SFEM_RESTRICT hx18,
        const real_t *const SFEM_RESTRICT hy18,
        const real_t *const SFEM_RESTRICT hz18,
        const real_t *const SFEM_RESTRICT hx19,
        const real_t *const SFEM_RESTRICT hy19,
        const real_t *const SFEM_RESTRICT hz19,
        const real_t *const SFEM_RESTRICT hx20,
        const real_t *const SFEM_RESTRICT hy20,
        const real_t *const SFEM_RESTRICT hz20,
        const real_t *const SFEM_RESTRICT hx21,
        const real_t *const SFEM_RESTRICT hy21,
        const real_t *const SFEM_RESTRICT hz21,
        const real_t *const SFEM_RESTRICT hx22,
        const real_t *const SFEM_RESTRICT hy22,
        const real_t *const SFEM_RESTRICT hz22,
        const real_t *const SFEM_RESTRICT hx23,
        const real_t *const SFEM_RESTRICT hy23,
        const real_t *const SFEM_RESTRICT hz23,
        const real_t *const SFEM_RESTRICT hx24,
        const real_t *const SFEM_RESTRICT hy24,
        const real_t *const SFEM_RESTRICT hz24,
        const real_t *const SFEM_RESTRICT hx25,
        const real_t *const SFEM_RESTRICT hy25,
        const real_t *const SFEM_RESTRICT hz25,
        const real_t *const SFEM_RESTRICT hx26,
        const real_t *const SFEM_RESTRICT hy26,
        const real_t *const SFEM_RESTRICT hz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_soa_impl<real_t, 27, 27, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, hx4, hy4, hz4, hx5, hy5, hz5, hx6, hy6, hz6, hx7, hy7, hz7, hx8, hy8, hz8, hx9, hy9, hz9, hx10, hy10, hz10, hx11, hy11, hz11, hx12, hy12, hz12, hx13, hy13, hz13, hx14, hy14, hz14, hx15, hy15, hz15, hx16, hy16, hz16, hx17, hy17, hz17, hx18, hy18, hz18, hx19, hy19, hz19, hx20, hy20, hz20, hx21, hy21, hz21, hx22, hy22, hz22, hx23, hy23, hz23, hx24, hy24, hz24, hx25, hy25, hz25, hx26, hy26, hz26, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9, outx10, outy10, outz10, outx11, outy11, outz11, outx12, outy12, outz12, outx13, outy13, outz13, outx14, outy14, outz14, outx15, outy15, outz15, outx16, outy16, outz16, outx17, outy17, outz17, outx18, outy18, outz18, outx19, outy19, outz19, outx20, outy20, outz20, outx21, outy21, outz21, outx22, outy22, outz22, outx23, outy23, outz23, outx24, outy24, outz24, outx25, outy25, outz25, outx26, outy26, outz26);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hz0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hz1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hz2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hz3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hz4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        const real_t *const SFEM_RESTRICT hz5,
        const real_t *const SFEM_RESTRICT hx6,
        const real_t *const SFEM_RESTRICT hy6,
        const real_t *const SFEM_RESTRICT hz6,
        const real_t *const SFEM_RESTRICT hx7,
        const real_t *const SFEM_RESTRICT hy7,
        const real_t *const SFEM_RESTRICT hz7,
        const real_t *const SFEM_RESTRICT hx8,
        const real_t *const SFEM_RESTRICT hy8,
        const real_t *const SFEM_RESTRICT hz8,
        const real_t *const SFEM_RESTRICT hx9,
        const real_t *const SFEM_RESTRICT hy9,
        const real_t *const SFEM_RESTRICT hz9,
        const real_t *const SFEM_RESTRICT hx10,
        const real_t *const SFEM_RESTRICT hy10,
        const real_t *const SFEM_RESTRICT hz10,
        const real_t *const SFEM_RESTRICT hx11,
        const real_t *const SFEM_RESTRICT hy11,
        const real_t *const SFEM_RESTRICT hz11,
        const real_t *const SFEM_RESTRICT hx12,
        const real_t *const SFEM_RESTRICT hy12,
        const real_t *const SFEM_RESTRICT hz12,
        const real_t *const SFEM_RESTRICT hx13,
        const real_t *const SFEM_RESTRICT hy13,
        const real_t *const SFEM_RESTRICT hz13,
        const real_t *const SFEM_RESTRICT hx14,
        const real_t *const SFEM_RESTRICT hy14,
        const real_t *const SFEM_RESTRICT hz14,
        const real_t *const SFEM_RESTRICT hx15,
        const real_t *const SFEM_RESTRICT hy15,
        const real_t *const SFEM_RESTRICT hz15,
        const real_t *const SFEM_RESTRICT hx16,
        const real_t *const SFEM_RESTRICT hy16,
        const real_t *const SFEM_RESTRICT hz16,
        const real_t *const SFEM_RESTRICT hx17,
        const real_t *const SFEM_RESTRICT hy17,
        const real_t *const SFEM_RESTRICT hz17,
        const real_t *const SFEM_RESTRICT hx18,
        const real_t *const SFEM_RESTRICT hy18,
        const real_t *const SFEM_RESTRICT hz18,
        const real_t *const SFEM_RESTRICT hx19,
        const real_t *const SFEM_RESTRICT hy19,
        const real_t *const SFEM_RESTRICT hz19,
        const real_t *const SFEM_RESTRICT hx20,
        const real_t *const SFEM_RESTRICT hy20,
        const real_t *const SFEM_RESTRICT hz20,
        const real_t *const SFEM_RESTRICT hx21,
        const real_t *const SFEM_RESTRICT hy21,
        const real_t *const SFEM_RESTRICT hz21,
        const real_t *const SFEM_RESTRICT hx22,
        const real_t *const SFEM_RESTRICT hy22,
        const real_t *const SFEM_RESTRICT hz22,
        const real_t *const SFEM_RESTRICT hx23,
        const real_t *const SFEM_RESTRICT hy23,
        const real_t *const SFEM_RESTRICT hz23,
        const real_t *const SFEM_RESTRICT hx24,
        const real_t *const SFEM_RESTRICT hy24,
        const real_t *const SFEM_RESTRICT hz24,
        const real_t *const SFEM_RESTRICT hx25,
        const real_t *const SFEM_RESTRICT hy25,
        const real_t *const SFEM_RESTRICT hz25,
        const real_t *const SFEM_RESTRICT hx26,
        const real_t *const SFEM_RESTRICT hy26,
        const real_t *const SFEM_RESTRICT hz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 27, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 27, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hz0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hz1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hz2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hz3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hz4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_hz5[VECTOR_SIZE];
        scalar_t block_hx6[VECTOR_SIZE];
        scalar_t block_hy6[VECTOR_SIZE];
        scalar_t block_hz6[VECTOR_SIZE];
        scalar_t block_hx7[VECTOR_SIZE];
        scalar_t block_hy7[VECTOR_SIZE];
        scalar_t block_hz7[VECTOR_SIZE];
        scalar_t block_hx8[VECTOR_SIZE];
        scalar_t block_hy8[VECTOR_SIZE];
        scalar_t block_hz8[VECTOR_SIZE];
        scalar_t block_hx9[VECTOR_SIZE];
        scalar_t block_hy9[VECTOR_SIZE];
        scalar_t block_hz9[VECTOR_SIZE];
        scalar_t block_hx10[VECTOR_SIZE];
        scalar_t block_hy10[VECTOR_SIZE];
        scalar_t block_hz10[VECTOR_SIZE];
        scalar_t block_hx11[VECTOR_SIZE];
        scalar_t block_hy11[VECTOR_SIZE];
        scalar_t block_hz11[VECTOR_SIZE];
        scalar_t block_hx12[VECTOR_SIZE];
        scalar_t block_hy12[VECTOR_SIZE];
        scalar_t block_hz12[VECTOR_SIZE];
        scalar_t block_hx13[VECTOR_SIZE];
        scalar_t block_hy13[VECTOR_SIZE];
        scalar_t block_hz13[VECTOR_SIZE];
        scalar_t block_hx14[VECTOR_SIZE];
        scalar_t block_hy14[VECTOR_SIZE];
        scalar_t block_hz14[VECTOR_SIZE];
        scalar_t block_hx15[VECTOR_SIZE];
        scalar_t block_hy15[VECTOR_SIZE];
        scalar_t block_hz15[VECTOR_SIZE];
        scalar_t block_hx16[VECTOR_SIZE];
        scalar_t block_hy16[VECTOR_SIZE];
        scalar_t block_hz16[VECTOR_SIZE];
        scalar_t block_hx17[VECTOR_SIZE];
        scalar_t block_hy17[VECTOR_SIZE];
        scalar_t block_hz17[VECTOR_SIZE];
        scalar_t block_hx18[VECTOR_SIZE];
        scalar_t block_hy18[VECTOR_SIZE];
        scalar_t block_hz18[VECTOR_SIZE];
        scalar_t block_hx19[VECTOR_SIZE];
        scalar_t block_hy19[VECTOR_SIZE];
        scalar_t block_hz19[VECTOR_SIZE];
        scalar_t block_hx20[VECTOR_SIZE];
        scalar_t block_hy20[VECTOR_SIZE];
        scalar_t block_hz20[VECTOR_SIZE];
        scalar_t block_hx21[VECTOR_SIZE];
        scalar_t block_hy21[VECTOR_SIZE];
        scalar_t block_hz21[VECTOR_SIZE];
        scalar_t block_hx22[VECTOR_SIZE];
        scalar_t block_hy22[VECTOR_SIZE];
        scalar_t block_hz22[VECTOR_SIZE];
        scalar_t block_hx23[VECTOR_SIZE];
        scalar_t block_hy23[VECTOR_SIZE];
        scalar_t block_hz23[VECTOR_SIZE];
        scalar_t block_hx24[VECTOR_SIZE];
        scalar_t block_hy24[VECTOR_SIZE];
        scalar_t block_hz24[VECTOR_SIZE];
        scalar_t block_hx25[VECTOR_SIZE];
        scalar_t block_hy25[VECTOR_SIZE];
        scalar_t block_hz25[VECTOR_SIZE];
        scalar_t block_hx26[VECTOR_SIZE];
        scalar_t block_hy26[VECTOR_SIZE];
        scalar_t block_hz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_z0[lane] = z0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_z1[lane] = z1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_z2[lane] = z2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_z3[lane] = z3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_z4[lane] = z4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_z5[lane] = z5[evbegin + lane];
            block_x6[lane] = x6[evbegin + lane];
            block_y6[lane] = y6[evbegin + lane];
            block_z6[lane] = z6[evbegin + lane];
            block_x7[lane] = x7[evbegin + lane];
            block_y7[lane] = y7[evbegin + lane];
            block_z7[lane] = z7[evbegin + lane];
            block_x8[lane] = x8[evbegin + lane];
            block_y8[lane] = y8[evbegin + lane];
            block_z8[lane] = z8[evbegin + lane];
            block_x9[lane] = x9[evbegin + lane];
            block_y9[lane] = y9[evbegin + lane];
            block_z9[lane] = z9[evbegin + lane];
            block_x10[lane] = x10[evbegin + lane];
            block_y10[lane] = y10[evbegin + lane];
            block_z10[lane] = z10[evbegin + lane];
            block_x11[lane] = x11[evbegin + lane];
            block_y11[lane] = y11[evbegin + lane];
            block_z11[lane] = z11[evbegin + lane];
            block_x12[lane] = x12[evbegin + lane];
            block_y12[lane] = y12[evbegin + lane];
            block_z12[lane] = z12[evbegin + lane];
            block_x13[lane] = x13[evbegin + lane];
            block_y13[lane] = y13[evbegin + lane];
            block_z13[lane] = z13[evbegin + lane];
            block_x14[lane] = x14[evbegin + lane];
            block_y14[lane] = y14[evbegin + lane];
            block_z14[lane] = z14[evbegin + lane];
            block_x15[lane] = x15[evbegin + lane];
            block_y15[lane] = y15[evbegin + lane];
            block_z15[lane] = z15[evbegin + lane];
            block_x16[lane] = x16[evbegin + lane];
            block_y16[lane] = y16[evbegin + lane];
            block_z16[lane] = z16[evbegin + lane];
            block_x17[lane] = x17[evbegin + lane];
            block_y17[lane] = y17[evbegin + lane];
            block_z17[lane] = z17[evbegin + lane];
            block_x18[lane] = x18[evbegin + lane];
            block_y18[lane] = y18[evbegin + lane];
            block_z18[lane] = z18[evbegin + lane];
            block_x19[lane] = x19[evbegin + lane];
            block_y19[lane] = y19[evbegin + lane];
            block_z19[lane] = z19[evbegin + lane];
            block_x20[lane] = x20[evbegin + lane];
            block_y20[lane] = y20[evbegin + lane];
            block_z20[lane] = z20[evbegin + lane];
            block_x21[lane] = x21[evbegin + lane];
            block_y21[lane] = y21[evbegin + lane];
            block_z21[lane] = z21[evbegin + lane];
            block_x22[lane] = x22[evbegin + lane];
            block_y22[lane] = y22[evbegin + lane];
            block_z22[lane] = z22[evbegin + lane];
            block_x23[lane] = x23[evbegin + lane];
            block_y23[lane] = y23[evbegin + lane];
            block_z23[lane] = z23[evbegin + lane];
            block_x24[lane] = x24[evbegin + lane];
            block_y24[lane] = y24[evbegin + lane];
            block_z24[lane] = z24[evbegin + lane];
            block_x25[lane] = x25[evbegin + lane];
            block_y25[lane] = y25[evbegin + lane];
            block_z25[lane] = z25[evbegin + lane];
            block_x26[lane] = x26[evbegin + lane];
            block_y26[lane] = y26[evbegin + lane];
            block_z26[lane] = z26[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_uz0[lane] = uz0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_uz1[lane] = uz1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_uz2[lane] = uz2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_uz3[lane] = uz3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_uz4[lane] = uz4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_uz5[lane] = uz5[evbegin + lane];
            block_ux6[lane] = ux6[evbegin + lane];
            block_uy6[lane] = uy6[evbegin + lane];
            block_uz6[lane] = uz6[evbegin + lane];
            block_ux7[lane] = ux7[evbegin + lane];
            block_uy7[lane] = uy7[evbegin + lane];
            block_uz7[lane] = uz7[evbegin + lane];
            block_ux8[lane] = ux8[evbegin + lane];
            block_uy8[lane] = uy8[evbegin + lane];
            block_uz8[lane] = uz8[evbegin + lane];
            block_ux9[lane] = ux9[evbegin + lane];
            block_uy9[lane] = uy9[evbegin + lane];
            block_uz9[lane] = uz9[evbegin + lane];
            block_ux10[lane] = ux10[evbegin + lane];
            block_uy10[lane] = uy10[evbegin + lane];
            block_uz10[lane] = uz10[evbegin + lane];
            block_ux11[lane] = ux11[evbegin + lane];
            block_uy11[lane] = uy11[evbegin + lane];
            block_uz11[lane] = uz11[evbegin + lane];
            block_ux12[lane] = ux12[evbegin + lane];
            block_uy12[lane] = uy12[evbegin + lane];
            block_uz12[lane] = uz12[evbegin + lane];
            block_ux13[lane] = ux13[evbegin + lane];
            block_uy13[lane] = uy13[evbegin + lane];
            block_uz13[lane] = uz13[evbegin + lane];
            block_ux14[lane] = ux14[evbegin + lane];
            block_uy14[lane] = uy14[evbegin + lane];
            block_uz14[lane] = uz14[evbegin + lane];
            block_ux15[lane] = ux15[evbegin + lane];
            block_uy15[lane] = uy15[evbegin + lane];
            block_uz15[lane] = uz15[evbegin + lane];
            block_ux16[lane] = ux16[evbegin + lane];
            block_uy16[lane] = uy16[evbegin + lane];
            block_uz16[lane] = uz16[evbegin + lane];
            block_ux17[lane] = ux17[evbegin + lane];
            block_uy17[lane] = uy17[evbegin + lane];
            block_uz17[lane] = uz17[evbegin + lane];
            block_ux18[lane] = ux18[evbegin + lane];
            block_uy18[lane] = uy18[evbegin + lane];
            block_uz18[lane] = uz18[evbegin + lane];
            block_ux19[lane] = ux19[evbegin + lane];
            block_uy19[lane] = uy19[evbegin + lane];
            block_uz19[lane] = uz19[evbegin + lane];
            block_ux20[lane] = ux20[evbegin + lane];
            block_uy20[lane] = uy20[evbegin + lane];
            block_uz20[lane] = uz20[evbegin + lane];
            block_ux21[lane] = ux21[evbegin + lane];
            block_uy21[lane] = uy21[evbegin + lane];
            block_uz21[lane] = uz21[evbegin + lane];
            block_ux22[lane] = ux22[evbegin + lane];
            block_uy22[lane] = uy22[evbegin + lane];
            block_uz22[lane] = uz22[evbegin + lane];
            block_ux23[lane] = ux23[evbegin + lane];
            block_uy23[lane] = uy23[evbegin + lane];
            block_uz23[lane] = uz23[evbegin + lane];
            block_ux24[lane] = ux24[evbegin + lane];
            block_uy24[lane] = uy24[evbegin + lane];
            block_uz24[lane] = uz24[evbegin + lane];
            block_ux25[lane] = ux25[evbegin + lane];
            block_uy25[lane] = uy25[evbegin + lane];
            block_uz25[lane] = uz25[evbegin + lane];
            block_ux26[lane] = ux26[evbegin + lane];
            block_uy26[lane] = uy26[evbegin + lane];
            block_uz26[lane] = uz26[evbegin + lane];
            block_hx0[lane] = hx0[evbegin + lane];
            block_hy0[lane] = hy0[evbegin + lane];
            block_hz0[lane] = hz0[evbegin + lane];
            block_hx1[lane] = hx1[evbegin + lane];
            block_hy1[lane] = hy1[evbegin + lane];
            block_hz1[lane] = hz1[evbegin + lane];
            block_hx2[lane] = hx2[evbegin + lane];
            block_hy2[lane] = hy2[evbegin + lane];
            block_hz2[lane] = hz2[evbegin + lane];
            block_hx3[lane] = hx3[evbegin + lane];
            block_hy3[lane] = hy3[evbegin + lane];
            block_hz3[lane] = hz3[evbegin + lane];
            block_hx4[lane] = hx4[evbegin + lane];
            block_hy4[lane] = hy4[evbegin + lane];
            block_hz4[lane] = hz4[evbegin + lane];
            block_hx5[lane] = hx5[evbegin + lane];
            block_hy5[lane] = hy5[evbegin + lane];
            block_hz5[lane] = hz5[evbegin + lane];
            block_hx6[lane] = hx6[evbegin + lane];
            block_hy6[lane] = hy6[evbegin + lane];
            block_hz6[lane] = hz6[evbegin + lane];
            block_hx7[lane] = hx7[evbegin + lane];
            block_hy7[lane] = hy7[evbegin + lane];
            block_hz7[lane] = hz7[evbegin + lane];
            block_hx8[lane] = hx8[evbegin + lane];
            block_hy8[lane] = hy8[evbegin + lane];
            block_hz8[lane] = hz8[evbegin + lane];
            block_hx9[lane] = hx9[evbegin + lane];
            block_hy9[lane] = hy9[evbegin + lane];
            block_hz9[lane] = hz9[evbegin + lane];
            block_hx10[lane] = hx10[evbegin + lane];
            block_hy10[lane] = hy10[evbegin + lane];
            block_hz10[lane] = hz10[evbegin + lane];
            block_hx11[lane] = hx11[evbegin + lane];
            block_hy11[lane] = hy11[evbegin + lane];
            block_hz11[lane] = hz11[evbegin + lane];
            block_hx12[lane] = hx12[evbegin + lane];
            block_hy12[lane] = hy12[evbegin + lane];
            block_hz12[lane] = hz12[evbegin + lane];
            block_hx13[lane] = hx13[evbegin + lane];
            block_hy13[lane] = hy13[evbegin + lane];
            block_hz13[lane] = hz13[evbegin + lane];
            block_hx14[lane] = hx14[evbegin + lane];
            block_hy14[lane] = hy14[evbegin + lane];
            block_hz14[lane] = hz14[evbegin + lane];
            block_hx15[lane] = hx15[evbegin + lane];
            block_hy15[lane] = hy15[evbegin + lane];
            block_hz15[lane] = hz15[evbegin + lane];
            block_hx16[lane] = hx16[evbegin + lane];
            block_hy16[lane] = hy16[evbegin + lane];
            block_hz16[lane] = hz16[evbegin + lane];
            block_hx17[lane] = hx17[evbegin + lane];
            block_hy17[lane] = hy17[evbegin + lane];
            block_hz17[lane] = hz17[evbegin + lane];
            block_hx18[lane] = hx18[evbegin + lane];
            block_hy18[lane] = hy18[evbegin + lane];
            block_hz18[lane] = hz18[evbegin + lane];
            block_hx19[lane] = hx19[evbegin + lane];
            block_hy19[lane] = hy19[evbegin + lane];
            block_hz19[lane] = hz19[evbegin + lane];
            block_hx20[lane] = hx20[evbegin + lane];
            block_hy20[lane] = hy20[evbegin + lane];
            block_hz20[lane] = hz20[evbegin + lane];
            block_hx21[lane] = hx21[evbegin + lane];
            block_hy21[lane] = hy21[evbegin + lane];
            block_hz21[lane] = hz21[evbegin + lane];
            block_hx22[lane] = hx22[evbegin + lane];
            block_hy22[lane] = hy22[evbegin + lane];
            block_hz22[lane] = hz22[evbegin + lane];
            block_hx23[lane] = hx23[evbegin + lane];
            block_hy23[lane] = hy23[evbegin + lane];
            block_hz23[lane] = hz23[evbegin + lane];
            block_hx24[lane] = hx24[evbegin + lane];
            block_hy24[lane] = hy24[evbegin + lane];
            block_hz24[lane] = hz24[evbegin + lane];
            block_hx25[lane] = hx25[evbegin + lane];
            block_hy25[lane] = hy25[evbegin + lane];
            block_hz25[lane] = hz25[evbegin + lane];
            block_hx26[lane] = hx26[evbegin + lane];
            block_hy26[lane] = hy26[evbegin + lane];
            block_hz26[lane] = hz26[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outz0[lane] = outz0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outz1[lane] = outz1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outz2[lane] = outz2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outz3[lane] = outz3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outz4[lane] = outz4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
            block_outz5[lane] = outz5[evbegin + lane];
            block_outx6[lane] = outx6[evbegin + lane];
            block_outy6[lane] = outy6[evbegin + lane];
            block_outz6[lane] = outz6[evbegin + lane];
            block_outx7[lane] = outx7[evbegin + lane];
            block_outy7[lane] = outy7[evbegin + lane];
            block_outz7[lane] = outz7[evbegin + lane];
            block_outx8[lane] = outx8[evbegin + lane];
            block_outy8[lane] = outy8[evbegin + lane];
            block_outz8[lane] = outz8[evbegin + lane];
            block_outx9[lane] = outx9[evbegin + lane];
            block_outy9[lane] = outy9[evbegin + lane];
            block_outz9[lane] = outz9[evbegin + lane];
            block_outx10[lane] = outx10[evbegin + lane];
            block_outy10[lane] = outy10[evbegin + lane];
            block_outz10[lane] = outz10[evbegin + lane];
            block_outx11[lane] = outx11[evbegin + lane];
            block_outy11[lane] = outy11[evbegin + lane];
            block_outz11[lane] = outz11[evbegin + lane];
            block_outx12[lane] = outx12[evbegin + lane];
            block_outy12[lane] = outy12[evbegin + lane];
            block_outz12[lane] = outz12[evbegin + lane];
            block_outx13[lane] = outx13[evbegin + lane];
            block_outy13[lane] = outy13[evbegin + lane];
            block_outz13[lane] = outz13[evbegin + lane];
            block_outx14[lane] = outx14[evbegin + lane];
            block_outy14[lane] = outy14[evbegin + lane];
            block_outz14[lane] = outz14[evbegin + lane];
            block_outx15[lane] = outx15[evbegin + lane];
            block_outy15[lane] = outy15[evbegin + lane];
            block_outz15[lane] = outz15[evbegin + lane];
            block_outx16[lane] = outx16[evbegin + lane];
            block_outy16[lane] = outy16[evbegin + lane];
            block_outz16[lane] = outz16[evbegin + lane];
            block_outx17[lane] = outx17[evbegin + lane];
            block_outy17[lane] = outy17[evbegin + lane];
            block_outz17[lane] = outz17[evbegin + lane];
            block_outx18[lane] = outx18[evbegin + lane];
            block_outy18[lane] = outy18[evbegin + lane];
            block_outz18[lane] = outz18[evbegin + lane];
            block_outx19[lane] = outx19[evbegin + lane];
            block_outy19[lane] = outy19[evbegin + lane];
            block_outz19[lane] = outz19[evbegin + lane];
            block_outx20[lane] = outx20[evbegin + lane];
            block_outy20[lane] = outy20[evbegin + lane];
            block_outz20[lane] = outz20[evbegin + lane];
            block_outx21[lane] = outx21[evbegin + lane];
            block_outy21[lane] = outy21[evbegin + lane];
            block_outz21[lane] = outz21[evbegin + lane];
            block_outx22[lane] = outx22[evbegin + lane];
            block_outy22[lane] = outy22[evbegin + lane];
            block_outz22[lane] = outz22[evbegin + lane];
            block_outx23[lane] = outx23[evbegin + lane];
            block_outy23[lane] = outy23[evbegin + lane];
            block_outz23[lane] = outz23[evbegin + lane];
            block_outx24[lane] = outx24[evbegin + lane];
            block_outy24[lane] = outy24[evbegin + lane];
            block_outz24[lane] = outz24[evbegin + lane];
            block_outx25[lane] = outx25[evbegin + lane];
            block_outy25[lane] = outy25[evbegin + lane];
            block_outz25[lane] = outz25[evbegin + lane];
            block_outx26[lane] = outx26[evbegin + lane];
            block_outy26[lane] = outy26[evbegin + lane];
            block_outz26[lane] = outz26[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx8, block_hy8, block_hz8, block_hx1, block_hy1, block_hz1, block_hx11, block_hy11, block_hz11, block_hx24, block_hy24, block_hz24, block_hx9, block_hy9, block_hz9, block_hx3, block_hy3, block_hz3, block_hx10, block_hy10, block_hz10, block_hx2, block_hy2, block_hz2, block_hx16, block_hy16, block_hz16, block_hx20, block_hy20, block_hz20, block_hx17, block_hy17, block_hz17, block_hx23, block_hy23, block_hz23, block_hx26, block_hy26, block_hz26, block_hx21, block_hy21, block_hz21, block_hx19, block_hy19, block_hz19, block_hx22, block_hy22, block_hz22, block_hx18, block_hy18, block_hz18, block_hx4, block_hy4, block_hz4, block_hx12, block_hy12, block_hz12, block_hx5, block_hy5, block_hz5, block_hx15, block_hy15, block_hz15, block_hx25, block_hy25, block_hz25, block_hx13, block_hy13, block_hz13, block_hx7, block_hy7, block_hz7, block_hx14, block_hy14, block_hz14, block_hx6, block_hy6, block_hz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outz0[evbegin + lane] = block_outz0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outz1[evbegin + lane] = block_outz1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outz2[evbegin + lane] = block_outz2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outz3[evbegin + lane] = block_outz3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outz4[evbegin + lane] = block_outz4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
            outz5[evbegin + lane] = block_outz5[lane];
            outx6[evbegin + lane] = block_outx6[lane];
            outy6[evbegin + lane] = block_outy6[lane];
            outz6[evbegin + lane] = block_outz6[lane];
            outx7[evbegin + lane] = block_outx7[lane];
            outy7[evbegin + lane] = block_outy7[lane];
            outz7[evbegin + lane] = block_outz7[lane];
            outx8[evbegin + lane] = block_outx8[lane];
            outy8[evbegin + lane] = block_outy8[lane];
            outz8[evbegin + lane] = block_outz8[lane];
            outx9[evbegin + lane] = block_outx9[lane];
            outy9[evbegin + lane] = block_outy9[lane];
            outz9[evbegin + lane] = block_outz9[lane];
            outx10[evbegin + lane] = block_outx10[lane];
            outy10[evbegin + lane] = block_outy10[lane];
            outz10[evbegin + lane] = block_outz10[lane];
            outx11[evbegin + lane] = block_outx11[lane];
            outy11[evbegin + lane] = block_outy11[lane];
            outz11[evbegin + lane] = block_outz11[lane];
            outx12[evbegin + lane] = block_outx12[lane];
            outy12[evbegin + lane] = block_outy12[lane];
            outz12[evbegin + lane] = block_outz12[lane];
            outx13[evbegin + lane] = block_outx13[lane];
            outy13[evbegin + lane] = block_outy13[lane];
            outz13[evbegin + lane] = block_outz13[lane];
            outx14[evbegin + lane] = block_outx14[lane];
            outy14[evbegin + lane] = block_outy14[lane];
            outz14[evbegin + lane] = block_outz14[lane];
            outx15[evbegin + lane] = block_outx15[lane];
            outy15[evbegin + lane] = block_outy15[lane];
            outz15[evbegin + lane] = block_outz15[lane];
            outx16[evbegin + lane] = block_outx16[lane];
            outy16[evbegin + lane] = block_outy16[lane];
            outz16[evbegin + lane] = block_outz16[lane];
            outx17[evbegin + lane] = block_outx17[lane];
            outy17[evbegin + lane] = block_outy17[lane];
            outz17[evbegin + lane] = block_outz17[lane];
            outx18[evbegin + lane] = block_outx18[lane];
            outy18[evbegin + lane] = block_outy18[lane];
            outz18[evbegin + lane] = block_outz18[lane];
            outx19[evbegin + lane] = block_outx19[lane];
            outy19[evbegin + lane] = block_outy19[lane];
            outz19[evbegin + lane] = block_outz19[lane];
            outx20[evbegin + lane] = block_outx20[lane];
            outy20[evbegin + lane] = block_outy20[lane];
            outz20[evbegin + lane] = block_outz20[lane];
            outx21[evbegin + lane] = block_outx21[lane];
            outy21[evbegin + lane] = block_outy21[lane];
            outz21[evbegin + lane] = block_outz21[lane];
            outx22[evbegin + lane] = block_outx22[lane];
            outy22[evbegin + lane] = block_outy22[lane];
            outz22[evbegin + lane] = block_outz22[lane];
            outx23[evbegin + lane] = block_outx23[lane];
            outy23[evbegin + lane] = block_outy23[lane];
            outz23[evbegin + lane] = block_outz23[lane];
            outx24[evbegin + lane] = block_outx24[lane];
            outy24[evbegin + lane] = block_outy24[lane];
            outz24[evbegin + lane] = block_outz24[lane];
            outx25[evbegin + lane] = block_outx25[lane];
            outy25[evbegin + lane] = block_outy25[lane];
            outz25[evbegin + lane] = block_outz25[lane];
            outx26[evbegin + lane] = block_outx26[lane];
            outy26[evbegin + lane] = block_outy26[lane];
            outz26[evbegin + lane] = block_outz26[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT z0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT z1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT z2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT z3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT z4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t *const SFEM_RESTRICT z5,
        const real_t *const SFEM_RESTRICT x6,
        const real_t *const SFEM_RESTRICT y6,
        const real_t *const SFEM_RESTRICT z6,
        const real_t *const SFEM_RESTRICT x7,
        const real_t *const SFEM_RESTRICT y7,
        const real_t *const SFEM_RESTRICT z7,
        const real_t *const SFEM_RESTRICT x8,
        const real_t *const SFEM_RESTRICT y8,
        const real_t *const SFEM_RESTRICT z8,
        const real_t *const SFEM_RESTRICT x9,
        const real_t *const SFEM_RESTRICT y9,
        const real_t *const SFEM_RESTRICT z9,
        const real_t *const SFEM_RESTRICT x10,
        const real_t *const SFEM_RESTRICT y10,
        const real_t *const SFEM_RESTRICT z10,
        const real_t *const SFEM_RESTRICT x11,
        const real_t *const SFEM_RESTRICT y11,
        const real_t *const SFEM_RESTRICT z11,
        const real_t *const SFEM_RESTRICT x12,
        const real_t *const SFEM_RESTRICT y12,
        const real_t *const SFEM_RESTRICT z12,
        const real_t *const SFEM_RESTRICT x13,
        const real_t *const SFEM_RESTRICT y13,
        const real_t *const SFEM_RESTRICT z13,
        const real_t *const SFEM_RESTRICT x14,
        const real_t *const SFEM_RESTRICT y14,
        const real_t *const SFEM_RESTRICT z14,
        const real_t *const SFEM_RESTRICT x15,
        const real_t *const SFEM_RESTRICT y15,
        const real_t *const SFEM_RESTRICT z15,
        const real_t *const SFEM_RESTRICT x16,
        const real_t *const SFEM_RESTRICT y16,
        const real_t *const SFEM_RESTRICT z16,
        const real_t *const SFEM_RESTRICT x17,
        const real_t *const SFEM_RESTRICT y17,
        const real_t *const SFEM_RESTRICT z17,
        const real_t *const SFEM_RESTRICT x18,
        const real_t *const SFEM_RESTRICT y18,
        const real_t *const SFEM_RESTRICT z18,
        const real_t *const SFEM_RESTRICT x19,
        const real_t *const SFEM_RESTRICT y19,
        const real_t *const SFEM_RESTRICT z19,
        const real_t *const SFEM_RESTRICT x20,
        const real_t *const SFEM_RESTRICT y20,
        const real_t *const SFEM_RESTRICT z20,
        const real_t *const SFEM_RESTRICT x21,
        const real_t *const SFEM_RESTRICT y21,
        const real_t *const SFEM_RESTRICT z21,
        const real_t *const SFEM_RESTRICT x22,
        const real_t *const SFEM_RESTRICT y22,
        const real_t *const SFEM_RESTRICT z22,
        const real_t *const SFEM_RESTRICT x23,
        const real_t *const SFEM_RESTRICT y23,
        const real_t *const SFEM_RESTRICT z23,
        const real_t *const SFEM_RESTRICT x24,
        const real_t *const SFEM_RESTRICT y24,
        const real_t *const SFEM_RESTRICT z24,
        const real_t *const SFEM_RESTRICT x25,
        const real_t *const SFEM_RESTRICT y25,
        const real_t *const SFEM_RESTRICT z25,
        const real_t *const SFEM_RESTRICT x26,
        const real_t *const SFEM_RESTRICT y26,
        const real_t *const SFEM_RESTRICT z26,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT uz0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT uz1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT uz2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT uz3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT uz4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT uz5,
        const real_t *const SFEM_RESTRICT ux6,
        const real_t *const SFEM_RESTRICT uy6,
        const real_t *const SFEM_RESTRICT uz6,
        const real_t *const SFEM_RESTRICT ux7,
        const real_t *const SFEM_RESTRICT uy7,
        const real_t *const SFEM_RESTRICT uz7,
        const real_t *const SFEM_RESTRICT ux8,
        const real_t *const SFEM_RESTRICT uy8,
        const real_t *const SFEM_RESTRICT uz8,
        const real_t *const SFEM_RESTRICT ux9,
        const real_t *const SFEM_RESTRICT uy9,
        const real_t *const SFEM_RESTRICT uz9,
        const real_t *const SFEM_RESTRICT ux10,
        const real_t *const SFEM_RESTRICT uy10,
        const real_t *const SFEM_RESTRICT uz10,
        const real_t *const SFEM_RESTRICT ux11,
        const real_t *const SFEM_RESTRICT uy11,
        const real_t *const SFEM_RESTRICT uz11,
        const real_t *const SFEM_RESTRICT ux12,
        const real_t *const SFEM_RESTRICT uy12,
        const real_t *const SFEM_RESTRICT uz12,
        const real_t *const SFEM_RESTRICT ux13,
        const real_t *const SFEM_RESTRICT uy13,
        const real_t *const SFEM_RESTRICT uz13,
        const real_t *const SFEM_RESTRICT ux14,
        const real_t *const SFEM_RESTRICT uy14,
        const real_t *const SFEM_RESTRICT uz14,
        const real_t *const SFEM_RESTRICT ux15,
        const real_t *const SFEM_RESTRICT uy15,
        const real_t *const SFEM_RESTRICT uz15,
        const real_t *const SFEM_RESTRICT ux16,
        const real_t *const SFEM_RESTRICT uy16,
        const real_t *const SFEM_RESTRICT uz16,
        const real_t *const SFEM_RESTRICT ux17,
        const real_t *const SFEM_RESTRICT uy17,
        const real_t *const SFEM_RESTRICT uz17,
        const real_t *const SFEM_RESTRICT ux18,
        const real_t *const SFEM_RESTRICT uy18,
        const real_t *const SFEM_RESTRICT uz18,
        const real_t *const SFEM_RESTRICT ux19,
        const real_t *const SFEM_RESTRICT uy19,
        const real_t *const SFEM_RESTRICT uz19,
        const real_t *const SFEM_RESTRICT ux20,
        const real_t *const SFEM_RESTRICT uy20,
        const real_t *const SFEM_RESTRICT uz20,
        const real_t *const SFEM_RESTRICT ux21,
        const real_t *const SFEM_RESTRICT uy21,
        const real_t *const SFEM_RESTRICT uz21,
        const real_t *const SFEM_RESTRICT ux22,
        const real_t *const SFEM_RESTRICT uy22,
        const real_t *const SFEM_RESTRICT uz22,
        const real_t *const SFEM_RESTRICT ux23,
        const real_t *const SFEM_RESTRICT uy23,
        const real_t *const SFEM_RESTRICT uz23,
        const real_t *const SFEM_RESTRICT ux24,
        const real_t *const SFEM_RESTRICT uy24,
        const real_t *const SFEM_RESTRICT uz24,
        const real_t *const SFEM_RESTRICT ux25,
        const real_t *const SFEM_RESTRICT uy25,
        const real_t *const SFEM_RESTRICT uz25,
        const real_t *const SFEM_RESTRICT ux26,
        const real_t *const SFEM_RESTRICT uy26,
        const real_t *const SFEM_RESTRICT uz26,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hz0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hz1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hz2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hz3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hz4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        const real_t *const SFEM_RESTRICT hz5,
        const real_t *const SFEM_RESTRICT hx6,
        const real_t *const SFEM_RESTRICT hy6,
        const real_t *const SFEM_RESTRICT hz6,
        const real_t *const SFEM_RESTRICT hx7,
        const real_t *const SFEM_RESTRICT hy7,
        const real_t *const SFEM_RESTRICT hz7,
        const real_t *const SFEM_RESTRICT hx8,
        const real_t *const SFEM_RESTRICT hy8,
        const real_t *const SFEM_RESTRICT hz8,
        const real_t *const SFEM_RESTRICT hx9,
        const real_t *const SFEM_RESTRICT hy9,
        const real_t *const SFEM_RESTRICT hz9,
        const real_t *const SFEM_RESTRICT hx10,
        const real_t *const SFEM_RESTRICT hy10,
        const real_t *const SFEM_RESTRICT hz10,
        const real_t *const SFEM_RESTRICT hx11,
        const real_t *const SFEM_RESTRICT hy11,
        const real_t *const SFEM_RESTRICT hz11,
        const real_t *const SFEM_RESTRICT hx12,
        const real_t *const SFEM_RESTRICT hy12,
        const real_t *const SFEM_RESTRICT hz12,
        const real_t *const SFEM_RESTRICT hx13,
        const real_t *const SFEM_RESTRICT hy13,
        const real_t *const SFEM_RESTRICT hz13,
        const real_t *const SFEM_RESTRICT hx14,
        const real_t *const SFEM_RESTRICT hy14,
        const real_t *const SFEM_RESTRICT hz14,
        const real_t *const SFEM_RESTRICT hx15,
        const real_t *const SFEM_RESTRICT hy15,
        const real_t *const SFEM_RESTRICT hz15,
        const real_t *const SFEM_RESTRICT hx16,
        const real_t *const SFEM_RESTRICT hy16,
        const real_t *const SFEM_RESTRICT hz16,
        const real_t *const SFEM_RESTRICT hx17,
        const real_t *const SFEM_RESTRICT hy17,
        const real_t *const SFEM_RESTRICT hz17,
        const real_t *const SFEM_RESTRICT hx18,
        const real_t *const SFEM_RESTRICT hy18,
        const real_t *const SFEM_RESTRICT hz18,
        const real_t *const SFEM_RESTRICT hx19,
        const real_t *const SFEM_RESTRICT hy19,
        const real_t *const SFEM_RESTRICT hz19,
        const real_t *const SFEM_RESTRICT hx20,
        const real_t *const SFEM_RESTRICT hy20,
        const real_t *const SFEM_RESTRICT hz20,
        const real_t *const SFEM_RESTRICT hx21,
        const real_t *const SFEM_RESTRICT hy21,
        const real_t *const SFEM_RESTRICT hz21,
        const real_t *const SFEM_RESTRICT hx22,
        const real_t *const SFEM_RESTRICT hy22,
        const real_t *const SFEM_RESTRICT hz22,
        const real_t *const SFEM_RESTRICT hx23,
        const real_t *const SFEM_RESTRICT hy23,
        const real_t *const SFEM_RESTRICT hz23,
        const real_t *const SFEM_RESTRICT hx24,
        const real_t *const SFEM_RESTRICT hy24,
        const real_t *const SFEM_RESTRICT hz24,
        const real_t *const SFEM_RESTRICT hx25,
        const real_t *const SFEM_RESTRICT hy25,
        const real_t *const SFEM_RESTRICT hz25,
        const real_t *const SFEM_RESTRICT hx26,
        const real_t *const SFEM_RESTRICT hy26,
        const real_t *const SFEM_RESTRICT hz26,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outz0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outz1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outz2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outz3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outz4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5,
        real_t *const SFEM_RESTRICT outz5,
        real_t *const SFEM_RESTRICT outx6,
        real_t *const SFEM_RESTRICT outy6,
        real_t *const SFEM_RESTRICT outz6,
        real_t *const SFEM_RESTRICT outx7,
        real_t *const SFEM_RESTRICT outy7,
        real_t *const SFEM_RESTRICT outz7,
        real_t *const SFEM_RESTRICT outx8,
        real_t *const SFEM_RESTRICT outy8,
        real_t *const SFEM_RESTRICT outz8,
        real_t *const SFEM_RESTRICT outx9,
        real_t *const SFEM_RESTRICT outy9,
        real_t *const SFEM_RESTRICT outz9,
        real_t *const SFEM_RESTRICT outx10,
        real_t *const SFEM_RESTRICT outy10,
        real_t *const SFEM_RESTRICT outz10,
        real_t *const SFEM_RESTRICT outx11,
        real_t *const SFEM_RESTRICT outy11,
        real_t *const SFEM_RESTRICT outz11,
        real_t *const SFEM_RESTRICT outx12,
        real_t *const SFEM_RESTRICT outy12,
        real_t *const SFEM_RESTRICT outz12,
        real_t *const SFEM_RESTRICT outx13,
        real_t *const SFEM_RESTRICT outy13,
        real_t *const SFEM_RESTRICT outz13,
        real_t *const SFEM_RESTRICT outx14,
        real_t *const SFEM_RESTRICT outy14,
        real_t *const SFEM_RESTRICT outz14,
        real_t *const SFEM_RESTRICT outx15,
        real_t *const SFEM_RESTRICT outy15,
        real_t *const SFEM_RESTRICT outz15,
        real_t *const SFEM_RESTRICT outx16,
        real_t *const SFEM_RESTRICT outy16,
        real_t *const SFEM_RESTRICT outz16,
        real_t *const SFEM_RESTRICT outx17,
        real_t *const SFEM_RESTRICT outy17,
        real_t *const SFEM_RESTRICT outz17,
        real_t *const SFEM_RESTRICT outx18,
        real_t *const SFEM_RESTRICT outy18,
        real_t *const SFEM_RESTRICT outz18,
        real_t *const SFEM_RESTRICT outx19,
        real_t *const SFEM_RESTRICT outy19,
        real_t *const SFEM_RESTRICT outz19,
        real_t *const SFEM_RESTRICT outx20,
        real_t *const SFEM_RESTRICT outy20,
        real_t *const SFEM_RESTRICT outz20,
        real_t *const SFEM_RESTRICT outx21,
        real_t *const SFEM_RESTRICT outy21,
        real_t *const SFEM_RESTRICT outz21,
        real_t *const SFEM_RESTRICT outx22,
        real_t *const SFEM_RESTRICT outy22,
        real_t *const SFEM_RESTRICT outz22,
        real_t *const SFEM_RESTRICT outx23,
        real_t *const SFEM_RESTRICT outy23,
        real_t *const SFEM_RESTRICT outz23,
        real_t *const SFEM_RESTRICT outx24,
        real_t *const SFEM_RESTRICT outy24,
        real_t *const SFEM_RESTRICT outz24,
        real_t *const SFEM_RESTRICT outx25,
        real_t *const SFEM_RESTRICT outy25,
        real_t *const SFEM_RESTRICT outz25,
        real_t *const SFEM_RESTRICT outx26,
        real_t *const SFEM_RESTRICT outy26,
        real_t *const SFEM_RESTRICT outz26
) {
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_isoparametric_soa_impl<real_t, 27, 27, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, x10, y10, z10, x11, y11, z11, x12, y12, z12, x13, y13, z13, x14, y14, z14, x15, y15, z15, x16, y16, z16, x17, y17, z17, x18, y18, z18, x19, y19, z19, x20, y20, z20, x21, y21, z21, x22, y22, z22, x23, y23, z23, x24, y24, z24, x25, y25, z25, x26, y26, z26, sfem::codegen::generated_neohookean_ogden_hex27_hex27_shape_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_grad_1d, sfem::codegen::generated_neohookean_ogden_hex27_hex27_q_weight_1d, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, ux10, uy10, uz10, ux11, uy11, uz11, ux12, uy12, uz12, ux13, uy13, uz13, ux14, uy14, uz14, ux15, uy15, uz15, ux16, uy16, uz16, ux17, uy17, uz17, ux18, uy18, uz18, ux19, uy19, uz19, ux20, uy20, uz20, ux21, uy21, uz21, ux22, uy22, uz22, ux23, uy23, uz23, ux24, uy24, uz24, ux25, uy25, uz25, ux26, uy26, uz26, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, hx4, hy4, hz4, hx5, hy5, hz5, hx6, hy6, hz6, hx7, hy7, hz7, hx8, hy8, hz8, hx9, hy9, hz9, hx10, hy10, hz10, hx11, hy11, hz11, hx12, hy12, hz12, hx13, hy13, hz13, hx14, hy14, hz14, hx15, hy15, hz15, hx16, hy16, hz16, hx17, hy17, hz17, hx18, hy18, hz18, hx19, hy19, hz19, hx20, hy20, hz20, hx21, hy21, hz21, hx22, hy22, hz22, hx23, hy23, hz23, hx24, hy24, hz24, hx25, hy25, hz25, hx26, hy26, hz26, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9, outx10, outy10, outz10, outx11, outy11, outz11, outx12, outy12, outz12, outx13, outy13, outz13, outx14, outy14, outz14, outx15, outy15, outz15, outx16, outy16, outz16, outx17, outy17, outz17, outx18, outy18, outz18, outx19, outy19, outz19, outx20, outy20, outz20, outx21, outy21, outz21, outx22, outy22, outz22, outx23, outy23, outz23, outx24, outy24, outz24, outx25, outy25, outz25, outx26, outy26, outz26);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hz0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hz1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hz2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hz3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hz4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_hz5[VECTOR_SIZE];
        scalar_t block_hx6[VECTOR_SIZE];
        scalar_t block_hy6[VECTOR_SIZE];
        scalar_t block_hz6[VECTOR_SIZE];
        scalar_t block_hx7[VECTOR_SIZE];
        scalar_t block_hy7[VECTOR_SIZE];
        scalar_t block_hz7[VECTOR_SIZE];
        scalar_t block_hx8[VECTOR_SIZE];
        scalar_t block_hy8[VECTOR_SIZE];
        scalar_t block_hz8[VECTOR_SIZE];
        scalar_t block_hx9[VECTOR_SIZE];
        scalar_t block_hy9[VECTOR_SIZE];
        scalar_t block_hz9[VECTOR_SIZE];
        scalar_t block_hx10[VECTOR_SIZE];
        scalar_t block_hy10[VECTOR_SIZE];
        scalar_t block_hz10[VECTOR_SIZE];
        scalar_t block_hx11[VECTOR_SIZE];
        scalar_t block_hy11[VECTOR_SIZE];
        scalar_t block_hz11[VECTOR_SIZE];
        scalar_t block_hx12[VECTOR_SIZE];
        scalar_t block_hy12[VECTOR_SIZE];
        scalar_t block_hz12[VECTOR_SIZE];
        scalar_t block_hx13[VECTOR_SIZE];
        scalar_t block_hy13[VECTOR_SIZE];
        scalar_t block_hz13[VECTOR_SIZE];
        scalar_t block_hx14[VECTOR_SIZE];
        scalar_t block_hy14[VECTOR_SIZE];
        scalar_t block_hz14[VECTOR_SIZE];
        scalar_t block_hx15[VECTOR_SIZE];
        scalar_t block_hy15[VECTOR_SIZE];
        scalar_t block_hz15[VECTOR_SIZE];
        scalar_t block_hx16[VECTOR_SIZE];
        scalar_t block_hy16[VECTOR_SIZE];
        scalar_t block_hz16[VECTOR_SIZE];
        scalar_t block_hx17[VECTOR_SIZE];
        scalar_t block_hy17[VECTOR_SIZE];
        scalar_t block_hz17[VECTOR_SIZE];
        scalar_t block_hx18[VECTOR_SIZE];
        scalar_t block_hy18[VECTOR_SIZE];
        scalar_t block_hz18[VECTOR_SIZE];
        scalar_t block_hx19[VECTOR_SIZE];
        scalar_t block_hy19[VECTOR_SIZE];
        scalar_t block_hz19[VECTOR_SIZE];
        scalar_t block_hx20[VECTOR_SIZE];
        scalar_t block_hy20[VECTOR_SIZE];
        scalar_t block_hz20[VECTOR_SIZE];
        scalar_t block_hx21[VECTOR_SIZE];
        scalar_t block_hy21[VECTOR_SIZE];
        scalar_t block_hz21[VECTOR_SIZE];
        scalar_t block_hx22[VECTOR_SIZE];
        scalar_t block_hy22[VECTOR_SIZE];
        scalar_t block_hz22[VECTOR_SIZE];
        scalar_t block_hx23[VECTOR_SIZE];
        scalar_t block_hy23[VECTOR_SIZE];
        scalar_t block_hz23[VECTOR_SIZE];
        scalar_t block_hx24[VECTOR_SIZE];
        scalar_t block_hy24[VECTOR_SIZE];
        scalar_t block_hz24[VECTOR_SIZE];
        scalar_t block_hx25[VECTOR_SIZE];
        scalar_t block_hy25[VECTOR_SIZE];
        scalar_t block_hz25[VECTOR_SIZE];
        scalar_t block_hx26[VECTOR_SIZE];
        scalar_t block_hy26[VECTOR_SIZE];
        scalar_t block_hz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_hx0[lane] = hx[ev[lane * N_SHAPE + 0] * h_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_hy0[lane] = hy[ev[lane * N_SHAPE + 0] * h_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_hz0[lane] = hz[ev[lane * N_SHAPE + 0] * h_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_hx1[lane] = hx[ev[lane * N_SHAPE + 1] * h_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_hy1[lane] = hy[ev[lane * N_SHAPE + 1] * h_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_hz1[lane] = hz[ev[lane * N_SHAPE + 1] * h_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_hx2[lane] = hx[ev[lane * N_SHAPE + 2] * h_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_hy2[lane] = hy[ev[lane * N_SHAPE + 2] * h_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_hz2[lane] = hz[ev[lane * N_SHAPE + 2] * h_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_hx3[lane] = hx[ev[lane * N_SHAPE + 3] * h_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_hy3[lane] = hy[ev[lane * N_SHAPE + 3] * h_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_hz3[lane] = hz[ev[lane * N_SHAPE + 3] * h_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_hx4[lane] = hx[ev[lane * N_SHAPE + 4] * h_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_hy4[lane] = hy[ev[lane * N_SHAPE + 4] * h_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_hz4[lane] = hz[ev[lane * N_SHAPE + 4] * h_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_hx5[lane] = hx[ev[lane * N_SHAPE + 5] * h_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_hy5[lane] = hy[ev[lane * N_SHAPE + 5] * h_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_hz5[lane] = hz[ev[lane * N_SHAPE + 5] * h_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_hx6[lane] = hx[ev[lane * N_SHAPE + 6] * h_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_hy6[lane] = hy[ev[lane * N_SHAPE + 6] * h_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_hz6[lane] = hz[ev[lane * N_SHAPE + 6] * h_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_hx7[lane] = hx[ev[lane * N_SHAPE + 7] * h_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_hy7[lane] = hy[ev[lane * N_SHAPE + 7] * h_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_hz7[lane] = hz[ev[lane * N_SHAPE + 7] * h_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_hx8[lane] = hx[ev[lane * N_SHAPE + 8] * h_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_hy8[lane] = hy[ev[lane * N_SHAPE + 8] * h_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_hz8[lane] = hz[ev[lane * N_SHAPE + 8] * h_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_hx9[lane] = hx[ev[lane * N_SHAPE + 9] * h_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_hy9[lane] = hy[ev[lane * N_SHAPE + 9] * h_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_hz9[lane] = hz[ev[lane * N_SHAPE + 9] * h_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_hx10[lane] = hx[ev[lane * N_SHAPE + 10] * h_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_hy10[lane] = hy[ev[lane * N_SHAPE + 10] * h_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_hz10[lane] = hz[ev[lane * N_SHAPE + 10] * h_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_hx11[lane] = hx[ev[lane * N_SHAPE + 11] * h_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_hy11[lane] = hy[ev[lane * N_SHAPE + 11] * h_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_hz11[lane] = hz[ev[lane * N_SHAPE + 11] * h_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_hx12[lane] = hx[ev[lane * N_SHAPE + 12] * h_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_hy12[lane] = hy[ev[lane * N_SHAPE + 12] * h_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_hz12[lane] = hz[ev[lane * N_SHAPE + 12] * h_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_hx13[lane] = hx[ev[lane * N_SHAPE + 13] * h_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_hy13[lane] = hy[ev[lane * N_SHAPE + 13] * h_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_hz13[lane] = hz[ev[lane * N_SHAPE + 13] * h_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_hx14[lane] = hx[ev[lane * N_SHAPE + 14] * h_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_hy14[lane] = hy[ev[lane * N_SHAPE + 14] * h_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_hz14[lane] = hz[ev[lane * N_SHAPE + 14] * h_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_hx15[lane] = hx[ev[lane * N_SHAPE + 15] * h_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_hy15[lane] = hy[ev[lane * N_SHAPE + 15] * h_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_hz15[lane] = hz[ev[lane * N_SHAPE + 15] * h_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_hx16[lane] = hx[ev[lane * N_SHAPE + 16] * h_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_hy16[lane] = hy[ev[lane * N_SHAPE + 16] * h_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_hz16[lane] = hz[ev[lane * N_SHAPE + 16] * h_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_hx17[lane] = hx[ev[lane * N_SHAPE + 17] * h_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_hy17[lane] = hy[ev[lane * N_SHAPE + 17] * h_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_hz17[lane] = hz[ev[lane * N_SHAPE + 17] * h_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_hx18[lane] = hx[ev[lane * N_SHAPE + 18] * h_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_hy18[lane] = hy[ev[lane * N_SHAPE + 18] * h_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_hz18[lane] = hz[ev[lane * N_SHAPE + 18] * h_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_hx19[lane] = hx[ev[lane * N_SHAPE + 19] * h_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_hy19[lane] = hy[ev[lane * N_SHAPE + 19] * h_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_hz19[lane] = hz[ev[lane * N_SHAPE + 19] * h_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_hx20[lane] = hx[ev[lane * N_SHAPE + 20] * h_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_hy20[lane] = hy[ev[lane * N_SHAPE + 20] * h_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_hz20[lane] = hz[ev[lane * N_SHAPE + 20] * h_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_hx21[lane] = hx[ev[lane * N_SHAPE + 21] * h_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_hy21[lane] = hy[ev[lane * N_SHAPE + 21] * h_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_hz21[lane] = hz[ev[lane * N_SHAPE + 21] * h_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_hx22[lane] = hx[ev[lane * N_SHAPE + 22] * h_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_hy22[lane] = hy[ev[lane * N_SHAPE + 22] * h_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_hz22[lane] = hz[ev[lane * N_SHAPE + 22] * h_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_hx23[lane] = hx[ev[lane * N_SHAPE + 23] * h_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_hy23[lane] = hy[ev[lane * N_SHAPE + 23] * h_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_hz23[lane] = hz[ev[lane * N_SHAPE + 23] * h_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_hx24[lane] = hx[ev[lane * N_SHAPE + 24] * h_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_hy24[lane] = hy[ev[lane * N_SHAPE + 24] * h_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_hz24[lane] = hz[ev[lane * N_SHAPE + 24] * h_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_hx25[lane] = hx[ev[lane * N_SHAPE + 25] * h_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_hy25[lane] = hy[ev[lane * N_SHAPE + 25] * h_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_hz25[lane] = hz[ev[lane * N_SHAPE + 25] * h_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_hx26[lane] = hx[ev[lane * N_SHAPE + 26] * h_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_hy26[lane] = hy[ev[lane * N_SHAPE + 26] * h_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_hz26[lane] = hz[ev[lane * N_SHAPE + 26] * h_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outz0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outz1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outz2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outz3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outz4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
            block_outz5[lane] = scalar_t(0);
            block_outx6[lane] = scalar_t(0);
            block_outy6[lane] = scalar_t(0);
            block_outz6[lane] = scalar_t(0);
            block_outx7[lane] = scalar_t(0);
            block_outy7[lane] = scalar_t(0);
            block_outz7[lane] = scalar_t(0);
            block_outx8[lane] = scalar_t(0);
            block_outy8[lane] = scalar_t(0);
            block_outz8[lane] = scalar_t(0);
            block_outx9[lane] = scalar_t(0);
            block_outy9[lane] = scalar_t(0);
            block_outz9[lane] = scalar_t(0);
            block_outx10[lane] = scalar_t(0);
            block_outy10[lane] = scalar_t(0);
            block_outz10[lane] = scalar_t(0);
            block_outx11[lane] = scalar_t(0);
            block_outy11[lane] = scalar_t(0);
            block_outz11[lane] = scalar_t(0);
            block_outx12[lane] = scalar_t(0);
            block_outy12[lane] = scalar_t(0);
            block_outz12[lane] = scalar_t(0);
            block_outx13[lane] = scalar_t(0);
            block_outy13[lane] = scalar_t(0);
            block_outz13[lane] = scalar_t(0);
            block_outx14[lane] = scalar_t(0);
            block_outy14[lane] = scalar_t(0);
            block_outz14[lane] = scalar_t(0);
            block_outx15[lane] = scalar_t(0);
            block_outy15[lane] = scalar_t(0);
            block_outz15[lane] = scalar_t(0);
            block_outx16[lane] = scalar_t(0);
            block_outy16[lane] = scalar_t(0);
            block_outz16[lane] = scalar_t(0);
            block_outx17[lane] = scalar_t(0);
            block_outy17[lane] = scalar_t(0);
            block_outz17[lane] = scalar_t(0);
            block_outx18[lane] = scalar_t(0);
            block_outy18[lane] = scalar_t(0);
            block_outz18[lane] = scalar_t(0);
            block_outx19[lane] = scalar_t(0);
            block_outy19[lane] = scalar_t(0);
            block_outz19[lane] = scalar_t(0);
            block_outx20[lane] = scalar_t(0);
            block_outy20[lane] = scalar_t(0);
            block_outz20[lane] = scalar_t(0);
            block_outx21[lane] = scalar_t(0);
            block_outy21[lane] = scalar_t(0);
            block_outz21[lane] = scalar_t(0);
            block_outx22[lane] = scalar_t(0);
            block_outy22[lane] = scalar_t(0);
            block_outz22[lane] = scalar_t(0);
            block_outx23[lane] = scalar_t(0);
            block_outy23[lane] = scalar_t(0);
            block_outz23[lane] = scalar_t(0);
            block_outx24[lane] = scalar_t(0);
            block_outy24[lane] = scalar_t(0);
            block_outz24[lane] = scalar_t(0);
            block_outx25[lane] = scalar_t(0);
            block_outy25[lane] = scalar_t(0);
            block_outz25[lane] = scalar_t(0);
            block_outx26[lane] = scalar_t(0);
            block_outy26[lane] = scalar_t(0);
            block_outz26[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx8, block_hy8, block_hz8, block_hx1, block_hy1, block_hz1, block_hx11, block_hy11, block_hz11, block_hx24, block_hy24, block_hz24, block_hx9, block_hy9, block_hz9, block_hx3, block_hy3, block_hz3, block_hx10, block_hy10, block_hz10, block_hx2, block_hy2, block_hz2, block_hx16, block_hy16, block_hz16, block_hx20, block_hy20, block_hz20, block_hx17, block_hy17, block_hz17, block_hx23, block_hy23, block_hz23, block_hx26, block_hy26, block_hz26, block_hx21, block_hy21, block_hz21, block_hx19, block_hy19, block_hz19, block_hx22, block_hy22, block_hz22, block_hx18, block_hy18, block_hz18, block_hx4, block_hy4, block_hz4, block_hx12, block_hy12, block_hz12, block_hx5, block_hy5, block_hz5, block_hx15, block_hy15, block_hz15, block_hx25, block_hy25, block_hz25, block_hx13, block_hy13, block_hz13, block_hx7, block_hy7, block_hz7, block_hx14, block_hy14, block_hz14, block_hx6, block_hy6, block_hz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        generated_neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 0] * out_stride] += block_outz0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 1] * out_stride] += block_outz1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 2] * out_stride] += block_outz2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 3] * out_stride] += block_outz3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 4] * out_stride] += block_outz4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 5] * out_stride] += block_outz5[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 6] * out_stride] += block_outx6[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 6] * out_stride] += block_outy6[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 6] * out_stride] += block_outz6[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 7] * out_stride] += block_outx7[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 7] * out_stride] += block_outy7[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 7] * out_stride] += block_outz7[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 8] * out_stride] += block_outx8[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 8] * out_stride] += block_outy8[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 8] * out_stride] += block_outz8[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 9] * out_stride] += block_outx9[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 9] * out_stride] += block_outy9[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 9] * out_stride] += block_outz9[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 10] * out_stride] += block_outx10[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 10] * out_stride] += block_outy10[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 10] * out_stride] += block_outz10[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 11] * out_stride] += block_outx11[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 11] * out_stride] += block_outy11[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 11] * out_stride] += block_outz11[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 12] * out_stride] += block_outx12[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 12] * out_stride] += block_outy12[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 12] * out_stride] += block_outz12[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 13] * out_stride] += block_outx13[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 13] * out_stride] += block_outy13[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 13] * out_stride] += block_outz13[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 14] * out_stride] += block_outx14[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 14] * out_stride] += block_outy14[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 14] * out_stride] += block_outz14[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 15] * out_stride] += block_outx15[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 15] * out_stride] += block_outy15[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 15] * out_stride] += block_outz15[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 16] * out_stride] += block_outx16[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 16] * out_stride] += block_outy16[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 16] * out_stride] += block_outz16[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 17] * out_stride] += block_outx17[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 17] * out_stride] += block_outy17[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 17] * out_stride] += block_outz17[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 18] * out_stride] += block_outx18[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 18] * out_stride] += block_outy18[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 18] * out_stride] += block_outz18[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 19] * out_stride] += block_outx19[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 19] * out_stride] += block_outy19[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 19] * out_stride] += block_outz19[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 20] * out_stride] += block_outx20[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 20] * out_stride] += block_outy20[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 20] * out_stride] += block_outz20[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 21] * out_stride] += block_outx21[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 21] * out_stride] += block_outy21[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 21] * out_stride] += block_outz21[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 22] * out_stride] += block_outx22[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 22] * out_stride] += block_outy22[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 22] * out_stride] += block_outz22[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 23] * out_stride] += block_outx23[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 23] * out_stride] += block_outy23[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 23] * out_stride] += block_outz23[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 24] * out_stride] += block_outx24[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 24] * out_stride] += block_outy24[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 24] * out_stride] += block_outz24[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 25] * out_stride] += block_outx25[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 25] * out_stride] += block_outy25[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 25] * out_stride] += block_outz25[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 26] * out_stride] += block_outx26[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 26] * out_stride] += block_outy26[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 26] * out_stride] += block_outz26[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t shape_1d[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
    static const scalar_t grad_1d[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
    static const scalar_t q_weight_1d[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_z0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_z1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_z2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_z3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_z4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_z5[VECTOR_SIZE];
        scalar_t block_x6[VECTOR_SIZE];
        scalar_t block_y6[VECTOR_SIZE];
        scalar_t block_z6[VECTOR_SIZE];
        scalar_t block_x7[VECTOR_SIZE];
        scalar_t block_y7[VECTOR_SIZE];
        scalar_t block_z7[VECTOR_SIZE];
        scalar_t block_x8[VECTOR_SIZE];
        scalar_t block_y8[VECTOR_SIZE];
        scalar_t block_z8[VECTOR_SIZE];
        scalar_t block_x9[VECTOR_SIZE];
        scalar_t block_y9[VECTOR_SIZE];
        scalar_t block_z9[VECTOR_SIZE];
        scalar_t block_x10[VECTOR_SIZE];
        scalar_t block_y10[VECTOR_SIZE];
        scalar_t block_z10[VECTOR_SIZE];
        scalar_t block_x11[VECTOR_SIZE];
        scalar_t block_y11[VECTOR_SIZE];
        scalar_t block_z11[VECTOR_SIZE];
        scalar_t block_x12[VECTOR_SIZE];
        scalar_t block_y12[VECTOR_SIZE];
        scalar_t block_z12[VECTOR_SIZE];
        scalar_t block_x13[VECTOR_SIZE];
        scalar_t block_y13[VECTOR_SIZE];
        scalar_t block_z13[VECTOR_SIZE];
        scalar_t block_x14[VECTOR_SIZE];
        scalar_t block_y14[VECTOR_SIZE];
        scalar_t block_z14[VECTOR_SIZE];
        scalar_t block_x15[VECTOR_SIZE];
        scalar_t block_y15[VECTOR_SIZE];
        scalar_t block_z15[VECTOR_SIZE];
        scalar_t block_x16[VECTOR_SIZE];
        scalar_t block_y16[VECTOR_SIZE];
        scalar_t block_z16[VECTOR_SIZE];
        scalar_t block_x17[VECTOR_SIZE];
        scalar_t block_y17[VECTOR_SIZE];
        scalar_t block_z17[VECTOR_SIZE];
        scalar_t block_x18[VECTOR_SIZE];
        scalar_t block_y18[VECTOR_SIZE];
        scalar_t block_z18[VECTOR_SIZE];
        scalar_t block_x19[VECTOR_SIZE];
        scalar_t block_y19[VECTOR_SIZE];
        scalar_t block_z19[VECTOR_SIZE];
        scalar_t block_x20[VECTOR_SIZE];
        scalar_t block_y20[VECTOR_SIZE];
        scalar_t block_z20[VECTOR_SIZE];
        scalar_t block_x21[VECTOR_SIZE];
        scalar_t block_y21[VECTOR_SIZE];
        scalar_t block_z21[VECTOR_SIZE];
        scalar_t block_x22[VECTOR_SIZE];
        scalar_t block_y22[VECTOR_SIZE];
        scalar_t block_z22[VECTOR_SIZE];
        scalar_t block_x23[VECTOR_SIZE];
        scalar_t block_y23[VECTOR_SIZE];
        scalar_t block_z23[VECTOR_SIZE];
        scalar_t block_x24[VECTOR_SIZE];
        scalar_t block_y24[VECTOR_SIZE];
        scalar_t block_z24[VECTOR_SIZE];
        scalar_t block_x25[VECTOR_SIZE];
        scalar_t block_y25[VECTOR_SIZE];
        scalar_t block_z25[VECTOR_SIZE];
        scalar_t block_x26[VECTOR_SIZE];
        scalar_t block_y26[VECTOR_SIZE];
        scalar_t block_z26[VECTOR_SIZE];
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
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_uz0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_uz1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_uz2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_uz3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_uz4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_uz5[VECTOR_SIZE];
        scalar_t block_ux6[VECTOR_SIZE];
        scalar_t block_uy6[VECTOR_SIZE];
        scalar_t block_uz6[VECTOR_SIZE];
        scalar_t block_ux7[VECTOR_SIZE];
        scalar_t block_uy7[VECTOR_SIZE];
        scalar_t block_uz7[VECTOR_SIZE];
        scalar_t block_ux8[VECTOR_SIZE];
        scalar_t block_uy8[VECTOR_SIZE];
        scalar_t block_uz8[VECTOR_SIZE];
        scalar_t block_ux9[VECTOR_SIZE];
        scalar_t block_uy9[VECTOR_SIZE];
        scalar_t block_uz9[VECTOR_SIZE];
        scalar_t block_ux10[VECTOR_SIZE];
        scalar_t block_uy10[VECTOR_SIZE];
        scalar_t block_uz10[VECTOR_SIZE];
        scalar_t block_ux11[VECTOR_SIZE];
        scalar_t block_uy11[VECTOR_SIZE];
        scalar_t block_uz11[VECTOR_SIZE];
        scalar_t block_ux12[VECTOR_SIZE];
        scalar_t block_uy12[VECTOR_SIZE];
        scalar_t block_uz12[VECTOR_SIZE];
        scalar_t block_ux13[VECTOR_SIZE];
        scalar_t block_uy13[VECTOR_SIZE];
        scalar_t block_uz13[VECTOR_SIZE];
        scalar_t block_ux14[VECTOR_SIZE];
        scalar_t block_uy14[VECTOR_SIZE];
        scalar_t block_uz14[VECTOR_SIZE];
        scalar_t block_ux15[VECTOR_SIZE];
        scalar_t block_uy15[VECTOR_SIZE];
        scalar_t block_uz15[VECTOR_SIZE];
        scalar_t block_ux16[VECTOR_SIZE];
        scalar_t block_uy16[VECTOR_SIZE];
        scalar_t block_uz16[VECTOR_SIZE];
        scalar_t block_ux17[VECTOR_SIZE];
        scalar_t block_uy17[VECTOR_SIZE];
        scalar_t block_uz17[VECTOR_SIZE];
        scalar_t block_ux18[VECTOR_SIZE];
        scalar_t block_uy18[VECTOR_SIZE];
        scalar_t block_uz18[VECTOR_SIZE];
        scalar_t block_ux19[VECTOR_SIZE];
        scalar_t block_uy19[VECTOR_SIZE];
        scalar_t block_uz19[VECTOR_SIZE];
        scalar_t block_ux20[VECTOR_SIZE];
        scalar_t block_uy20[VECTOR_SIZE];
        scalar_t block_uz20[VECTOR_SIZE];
        scalar_t block_ux21[VECTOR_SIZE];
        scalar_t block_uy21[VECTOR_SIZE];
        scalar_t block_uz21[VECTOR_SIZE];
        scalar_t block_ux22[VECTOR_SIZE];
        scalar_t block_uy22[VECTOR_SIZE];
        scalar_t block_uz22[VECTOR_SIZE];
        scalar_t block_ux23[VECTOR_SIZE];
        scalar_t block_uy23[VECTOR_SIZE];
        scalar_t block_uz23[VECTOR_SIZE];
        scalar_t block_ux24[VECTOR_SIZE];
        scalar_t block_uy24[VECTOR_SIZE];
        scalar_t block_uz24[VECTOR_SIZE];
        scalar_t block_ux25[VECTOR_SIZE];
        scalar_t block_uy25[VECTOR_SIZE];
        scalar_t block_uz25[VECTOR_SIZE];
        scalar_t block_ux26[VECTOR_SIZE];
        scalar_t block_uy26[VECTOR_SIZE];
        scalar_t block_uz26[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hz0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hz1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hz2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hz3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hz4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_hz5[VECTOR_SIZE];
        scalar_t block_hx6[VECTOR_SIZE];
        scalar_t block_hy6[VECTOR_SIZE];
        scalar_t block_hz6[VECTOR_SIZE];
        scalar_t block_hx7[VECTOR_SIZE];
        scalar_t block_hy7[VECTOR_SIZE];
        scalar_t block_hz7[VECTOR_SIZE];
        scalar_t block_hx8[VECTOR_SIZE];
        scalar_t block_hy8[VECTOR_SIZE];
        scalar_t block_hz8[VECTOR_SIZE];
        scalar_t block_hx9[VECTOR_SIZE];
        scalar_t block_hy9[VECTOR_SIZE];
        scalar_t block_hz9[VECTOR_SIZE];
        scalar_t block_hx10[VECTOR_SIZE];
        scalar_t block_hy10[VECTOR_SIZE];
        scalar_t block_hz10[VECTOR_SIZE];
        scalar_t block_hx11[VECTOR_SIZE];
        scalar_t block_hy11[VECTOR_SIZE];
        scalar_t block_hz11[VECTOR_SIZE];
        scalar_t block_hx12[VECTOR_SIZE];
        scalar_t block_hy12[VECTOR_SIZE];
        scalar_t block_hz12[VECTOR_SIZE];
        scalar_t block_hx13[VECTOR_SIZE];
        scalar_t block_hy13[VECTOR_SIZE];
        scalar_t block_hz13[VECTOR_SIZE];
        scalar_t block_hx14[VECTOR_SIZE];
        scalar_t block_hy14[VECTOR_SIZE];
        scalar_t block_hz14[VECTOR_SIZE];
        scalar_t block_hx15[VECTOR_SIZE];
        scalar_t block_hy15[VECTOR_SIZE];
        scalar_t block_hz15[VECTOR_SIZE];
        scalar_t block_hx16[VECTOR_SIZE];
        scalar_t block_hy16[VECTOR_SIZE];
        scalar_t block_hz16[VECTOR_SIZE];
        scalar_t block_hx17[VECTOR_SIZE];
        scalar_t block_hy17[VECTOR_SIZE];
        scalar_t block_hz17[VECTOR_SIZE];
        scalar_t block_hx18[VECTOR_SIZE];
        scalar_t block_hy18[VECTOR_SIZE];
        scalar_t block_hz18[VECTOR_SIZE];
        scalar_t block_hx19[VECTOR_SIZE];
        scalar_t block_hy19[VECTOR_SIZE];
        scalar_t block_hz19[VECTOR_SIZE];
        scalar_t block_hx20[VECTOR_SIZE];
        scalar_t block_hy20[VECTOR_SIZE];
        scalar_t block_hz20[VECTOR_SIZE];
        scalar_t block_hx21[VECTOR_SIZE];
        scalar_t block_hy21[VECTOR_SIZE];
        scalar_t block_hz21[VECTOR_SIZE];
        scalar_t block_hx22[VECTOR_SIZE];
        scalar_t block_hy22[VECTOR_SIZE];
        scalar_t block_hz22[VECTOR_SIZE];
        scalar_t block_hx23[VECTOR_SIZE];
        scalar_t block_hy23[VECTOR_SIZE];
        scalar_t block_hz23[VECTOR_SIZE];
        scalar_t block_hx24[VECTOR_SIZE];
        scalar_t block_hy24[VECTOR_SIZE];
        scalar_t block_hz24[VECTOR_SIZE];
        scalar_t block_hx25[VECTOR_SIZE];
        scalar_t block_hy25[VECTOR_SIZE];
        scalar_t block_hz25[VECTOR_SIZE];
        scalar_t block_hx26[VECTOR_SIZE];
        scalar_t block_hy26[VECTOR_SIZE];
        scalar_t block_hz26[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outz0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outz1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outz2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outz3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outz4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];
        scalar_t block_outz5[VECTOR_SIZE];
        scalar_t block_outx6[VECTOR_SIZE];
        scalar_t block_outy6[VECTOR_SIZE];
        scalar_t block_outz6[VECTOR_SIZE];
        scalar_t block_outx7[VECTOR_SIZE];
        scalar_t block_outy7[VECTOR_SIZE];
        scalar_t block_outz7[VECTOR_SIZE];
        scalar_t block_outx8[VECTOR_SIZE];
        scalar_t block_outy8[VECTOR_SIZE];
        scalar_t block_outz8[VECTOR_SIZE];
        scalar_t block_outx9[VECTOR_SIZE];
        scalar_t block_outy9[VECTOR_SIZE];
        scalar_t block_outz9[VECTOR_SIZE];
        scalar_t block_outx10[VECTOR_SIZE];
        scalar_t block_outy10[VECTOR_SIZE];
        scalar_t block_outz10[VECTOR_SIZE];
        scalar_t block_outx11[VECTOR_SIZE];
        scalar_t block_outy11[VECTOR_SIZE];
        scalar_t block_outz11[VECTOR_SIZE];
        scalar_t block_outx12[VECTOR_SIZE];
        scalar_t block_outy12[VECTOR_SIZE];
        scalar_t block_outz12[VECTOR_SIZE];
        scalar_t block_outx13[VECTOR_SIZE];
        scalar_t block_outy13[VECTOR_SIZE];
        scalar_t block_outz13[VECTOR_SIZE];
        scalar_t block_outx14[VECTOR_SIZE];
        scalar_t block_outy14[VECTOR_SIZE];
        scalar_t block_outz14[VECTOR_SIZE];
        scalar_t block_outx15[VECTOR_SIZE];
        scalar_t block_outy15[VECTOR_SIZE];
        scalar_t block_outz15[VECTOR_SIZE];
        scalar_t block_outx16[VECTOR_SIZE];
        scalar_t block_outy16[VECTOR_SIZE];
        scalar_t block_outz16[VECTOR_SIZE];
        scalar_t block_outx17[VECTOR_SIZE];
        scalar_t block_outy17[VECTOR_SIZE];
        scalar_t block_outz17[VECTOR_SIZE];
        scalar_t block_outx18[VECTOR_SIZE];
        scalar_t block_outy18[VECTOR_SIZE];
        scalar_t block_outz18[VECTOR_SIZE];
        scalar_t block_outx19[VECTOR_SIZE];
        scalar_t block_outy19[VECTOR_SIZE];
        scalar_t block_outz19[VECTOR_SIZE];
        scalar_t block_outx20[VECTOR_SIZE];
        scalar_t block_outy20[VECTOR_SIZE];
        scalar_t block_outz20[VECTOR_SIZE];
        scalar_t block_outx21[VECTOR_SIZE];
        scalar_t block_outy21[VECTOR_SIZE];
        scalar_t block_outz21[VECTOR_SIZE];
        scalar_t block_outx22[VECTOR_SIZE];
        scalar_t block_outy22[VECTOR_SIZE];
        scalar_t block_outz22[VECTOR_SIZE];
        scalar_t block_outx23[VECTOR_SIZE];
        scalar_t block_outy23[VECTOR_SIZE];
        scalar_t block_outz23[VECTOR_SIZE];
        scalar_t block_outx24[VECTOR_SIZE];
        scalar_t block_outy24[VECTOR_SIZE];
        scalar_t block_outz24[VECTOR_SIZE];
        scalar_t block_outx25[VECTOR_SIZE];
        scalar_t block_outy25[VECTOR_SIZE];
        scalar_t block_outz25[VECTOR_SIZE];
        scalar_t block_outx26[VECTOR_SIZE];
        scalar_t block_outy26[VECTOR_SIZE];
        scalar_t block_outz26[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_z0[lane] = z[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_z1[lane] = z[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_z2[lane] = z[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_z3[lane] = z[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_z4[lane] = z[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
            block_z5[lane] = z[ev[lane * N_SHAPE + 5]];
            block_x6[lane] = x[ev[lane * N_SHAPE + 6]];
            block_y6[lane] = y[ev[lane * N_SHAPE + 6]];
            block_z6[lane] = z[ev[lane * N_SHAPE + 6]];
            block_x7[lane] = x[ev[lane * N_SHAPE + 7]];
            block_y7[lane] = y[ev[lane * N_SHAPE + 7]];
            block_z7[lane] = z[ev[lane * N_SHAPE + 7]];
            block_x8[lane] = x[ev[lane * N_SHAPE + 8]];
            block_y8[lane] = y[ev[lane * N_SHAPE + 8]];
            block_z8[lane] = z[ev[lane * N_SHAPE + 8]];
            block_x9[lane] = x[ev[lane * N_SHAPE + 9]];
            block_y9[lane] = y[ev[lane * N_SHAPE + 9]];
            block_z9[lane] = z[ev[lane * N_SHAPE + 9]];
            block_x10[lane] = x[ev[lane * N_SHAPE + 10]];
            block_y10[lane] = y[ev[lane * N_SHAPE + 10]];
            block_z10[lane] = z[ev[lane * N_SHAPE + 10]];
            block_x11[lane] = x[ev[lane * N_SHAPE + 11]];
            block_y11[lane] = y[ev[lane * N_SHAPE + 11]];
            block_z11[lane] = z[ev[lane * N_SHAPE + 11]];
            block_x12[lane] = x[ev[lane * N_SHAPE + 12]];
            block_y12[lane] = y[ev[lane * N_SHAPE + 12]];
            block_z12[lane] = z[ev[lane * N_SHAPE + 12]];
            block_x13[lane] = x[ev[lane * N_SHAPE + 13]];
            block_y13[lane] = y[ev[lane * N_SHAPE + 13]];
            block_z13[lane] = z[ev[lane * N_SHAPE + 13]];
            block_x14[lane] = x[ev[lane * N_SHAPE + 14]];
            block_y14[lane] = y[ev[lane * N_SHAPE + 14]];
            block_z14[lane] = z[ev[lane * N_SHAPE + 14]];
            block_x15[lane] = x[ev[lane * N_SHAPE + 15]];
            block_y15[lane] = y[ev[lane * N_SHAPE + 15]];
            block_z15[lane] = z[ev[lane * N_SHAPE + 15]];
            block_x16[lane] = x[ev[lane * N_SHAPE + 16]];
            block_y16[lane] = y[ev[lane * N_SHAPE + 16]];
            block_z16[lane] = z[ev[lane * N_SHAPE + 16]];
            block_x17[lane] = x[ev[lane * N_SHAPE + 17]];
            block_y17[lane] = y[ev[lane * N_SHAPE + 17]];
            block_z17[lane] = z[ev[lane * N_SHAPE + 17]];
            block_x18[lane] = x[ev[lane * N_SHAPE + 18]];
            block_y18[lane] = y[ev[lane * N_SHAPE + 18]];
            block_z18[lane] = z[ev[lane * N_SHAPE + 18]];
            block_x19[lane] = x[ev[lane * N_SHAPE + 19]];
            block_y19[lane] = y[ev[lane * N_SHAPE + 19]];
            block_z19[lane] = z[ev[lane * N_SHAPE + 19]];
            block_x20[lane] = x[ev[lane * N_SHAPE + 20]];
            block_y20[lane] = y[ev[lane * N_SHAPE + 20]];
            block_z20[lane] = z[ev[lane * N_SHAPE + 20]];
            block_x21[lane] = x[ev[lane * N_SHAPE + 21]];
            block_y21[lane] = y[ev[lane * N_SHAPE + 21]];
            block_z21[lane] = z[ev[lane * N_SHAPE + 21]];
            block_x22[lane] = x[ev[lane * N_SHAPE + 22]];
            block_y22[lane] = y[ev[lane * N_SHAPE + 22]];
            block_z22[lane] = z[ev[lane * N_SHAPE + 22]];
            block_x23[lane] = x[ev[lane * N_SHAPE + 23]];
            block_y23[lane] = y[ev[lane * N_SHAPE + 23]];
            block_z23[lane] = z[ev[lane * N_SHAPE + 23]];
            block_x24[lane] = x[ev[lane * N_SHAPE + 24]];
            block_y24[lane] = y[ev[lane * N_SHAPE + 24]];
            block_z24[lane] = z[ev[lane * N_SHAPE + 24]];
            block_x25[lane] = x[ev[lane * N_SHAPE + 25]];
            block_y25[lane] = y[ev[lane * N_SHAPE + 25]];
            block_z25[lane] = z[ev[lane * N_SHAPE + 25]];
            block_x26[lane] = x[ev[lane * N_SHAPE + 26]];
            block_y26[lane] = y[ev[lane * N_SHAPE + 26]];
            block_z26[lane] = z[ev[lane * N_SHAPE + 26]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_hx0[lane] = hx[ev[lane * N_SHAPE + 0] * h_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_hy0[lane] = hy[ev[lane * N_SHAPE + 0] * h_stride];
            block_uz0[lane] = uz[ev[lane * N_SHAPE + 0] * u_stride];
            block_hz0[lane] = hz[ev[lane * N_SHAPE + 0] * h_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_hx1[lane] = hx[ev[lane * N_SHAPE + 1] * h_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_hy1[lane] = hy[ev[lane * N_SHAPE + 1] * h_stride];
            block_uz1[lane] = uz[ev[lane * N_SHAPE + 1] * u_stride];
            block_hz1[lane] = hz[ev[lane * N_SHAPE + 1] * h_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_hx2[lane] = hx[ev[lane * N_SHAPE + 2] * h_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_hy2[lane] = hy[ev[lane * N_SHAPE + 2] * h_stride];
            block_uz2[lane] = uz[ev[lane * N_SHAPE + 2] * u_stride];
            block_hz2[lane] = hz[ev[lane * N_SHAPE + 2] * h_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_hx3[lane] = hx[ev[lane * N_SHAPE + 3] * h_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_hy3[lane] = hy[ev[lane * N_SHAPE + 3] * h_stride];
            block_uz3[lane] = uz[ev[lane * N_SHAPE + 3] * u_stride];
            block_hz3[lane] = hz[ev[lane * N_SHAPE + 3] * h_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_hx4[lane] = hx[ev[lane * N_SHAPE + 4] * h_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_hy4[lane] = hy[ev[lane * N_SHAPE + 4] * h_stride];
            block_uz4[lane] = uz[ev[lane * N_SHAPE + 4] * u_stride];
            block_hz4[lane] = hz[ev[lane * N_SHAPE + 4] * h_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_hx5[lane] = hx[ev[lane * N_SHAPE + 5] * h_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_hy5[lane] = hy[ev[lane * N_SHAPE + 5] * h_stride];
            block_uz5[lane] = uz[ev[lane * N_SHAPE + 5] * u_stride];
            block_hz5[lane] = hz[ev[lane * N_SHAPE + 5] * h_stride];
            block_ux6[lane] = ux[ev[lane * N_SHAPE + 6] * u_stride];
            block_hx6[lane] = hx[ev[lane * N_SHAPE + 6] * h_stride];
            block_uy6[lane] = uy[ev[lane * N_SHAPE + 6] * u_stride];
            block_hy6[lane] = hy[ev[lane * N_SHAPE + 6] * h_stride];
            block_uz6[lane] = uz[ev[lane * N_SHAPE + 6] * u_stride];
            block_hz6[lane] = hz[ev[lane * N_SHAPE + 6] * h_stride];
            block_ux7[lane] = ux[ev[lane * N_SHAPE + 7] * u_stride];
            block_hx7[lane] = hx[ev[lane * N_SHAPE + 7] * h_stride];
            block_uy7[lane] = uy[ev[lane * N_SHAPE + 7] * u_stride];
            block_hy7[lane] = hy[ev[lane * N_SHAPE + 7] * h_stride];
            block_uz7[lane] = uz[ev[lane * N_SHAPE + 7] * u_stride];
            block_hz7[lane] = hz[ev[lane * N_SHAPE + 7] * h_stride];
            block_ux8[lane] = ux[ev[lane * N_SHAPE + 8] * u_stride];
            block_hx8[lane] = hx[ev[lane * N_SHAPE + 8] * h_stride];
            block_uy8[lane] = uy[ev[lane * N_SHAPE + 8] * u_stride];
            block_hy8[lane] = hy[ev[lane * N_SHAPE + 8] * h_stride];
            block_uz8[lane] = uz[ev[lane * N_SHAPE + 8] * u_stride];
            block_hz8[lane] = hz[ev[lane * N_SHAPE + 8] * h_stride];
            block_ux9[lane] = ux[ev[lane * N_SHAPE + 9] * u_stride];
            block_hx9[lane] = hx[ev[lane * N_SHAPE + 9] * h_stride];
            block_uy9[lane] = uy[ev[lane * N_SHAPE + 9] * u_stride];
            block_hy9[lane] = hy[ev[lane * N_SHAPE + 9] * h_stride];
            block_uz9[lane] = uz[ev[lane * N_SHAPE + 9] * u_stride];
            block_hz9[lane] = hz[ev[lane * N_SHAPE + 9] * h_stride];
            block_ux10[lane] = ux[ev[lane * N_SHAPE + 10] * u_stride];
            block_hx10[lane] = hx[ev[lane * N_SHAPE + 10] * h_stride];
            block_uy10[lane] = uy[ev[lane * N_SHAPE + 10] * u_stride];
            block_hy10[lane] = hy[ev[lane * N_SHAPE + 10] * h_stride];
            block_uz10[lane] = uz[ev[lane * N_SHAPE + 10] * u_stride];
            block_hz10[lane] = hz[ev[lane * N_SHAPE + 10] * h_stride];
            block_ux11[lane] = ux[ev[lane * N_SHAPE + 11] * u_stride];
            block_hx11[lane] = hx[ev[lane * N_SHAPE + 11] * h_stride];
            block_uy11[lane] = uy[ev[lane * N_SHAPE + 11] * u_stride];
            block_hy11[lane] = hy[ev[lane * N_SHAPE + 11] * h_stride];
            block_uz11[lane] = uz[ev[lane * N_SHAPE + 11] * u_stride];
            block_hz11[lane] = hz[ev[lane * N_SHAPE + 11] * h_stride];
            block_ux12[lane] = ux[ev[lane * N_SHAPE + 12] * u_stride];
            block_hx12[lane] = hx[ev[lane * N_SHAPE + 12] * h_stride];
            block_uy12[lane] = uy[ev[lane * N_SHAPE + 12] * u_stride];
            block_hy12[lane] = hy[ev[lane * N_SHAPE + 12] * h_stride];
            block_uz12[lane] = uz[ev[lane * N_SHAPE + 12] * u_stride];
            block_hz12[lane] = hz[ev[lane * N_SHAPE + 12] * h_stride];
            block_ux13[lane] = ux[ev[lane * N_SHAPE + 13] * u_stride];
            block_hx13[lane] = hx[ev[lane * N_SHAPE + 13] * h_stride];
            block_uy13[lane] = uy[ev[lane * N_SHAPE + 13] * u_stride];
            block_hy13[lane] = hy[ev[lane * N_SHAPE + 13] * h_stride];
            block_uz13[lane] = uz[ev[lane * N_SHAPE + 13] * u_stride];
            block_hz13[lane] = hz[ev[lane * N_SHAPE + 13] * h_stride];
            block_ux14[lane] = ux[ev[lane * N_SHAPE + 14] * u_stride];
            block_hx14[lane] = hx[ev[lane * N_SHAPE + 14] * h_stride];
            block_uy14[lane] = uy[ev[lane * N_SHAPE + 14] * u_stride];
            block_hy14[lane] = hy[ev[lane * N_SHAPE + 14] * h_stride];
            block_uz14[lane] = uz[ev[lane * N_SHAPE + 14] * u_stride];
            block_hz14[lane] = hz[ev[lane * N_SHAPE + 14] * h_stride];
            block_ux15[lane] = ux[ev[lane * N_SHAPE + 15] * u_stride];
            block_hx15[lane] = hx[ev[lane * N_SHAPE + 15] * h_stride];
            block_uy15[lane] = uy[ev[lane * N_SHAPE + 15] * u_stride];
            block_hy15[lane] = hy[ev[lane * N_SHAPE + 15] * h_stride];
            block_uz15[lane] = uz[ev[lane * N_SHAPE + 15] * u_stride];
            block_hz15[lane] = hz[ev[lane * N_SHAPE + 15] * h_stride];
            block_ux16[lane] = ux[ev[lane * N_SHAPE + 16] * u_stride];
            block_hx16[lane] = hx[ev[lane * N_SHAPE + 16] * h_stride];
            block_uy16[lane] = uy[ev[lane * N_SHAPE + 16] * u_stride];
            block_hy16[lane] = hy[ev[lane * N_SHAPE + 16] * h_stride];
            block_uz16[lane] = uz[ev[lane * N_SHAPE + 16] * u_stride];
            block_hz16[lane] = hz[ev[lane * N_SHAPE + 16] * h_stride];
            block_ux17[lane] = ux[ev[lane * N_SHAPE + 17] * u_stride];
            block_hx17[lane] = hx[ev[lane * N_SHAPE + 17] * h_stride];
            block_uy17[lane] = uy[ev[lane * N_SHAPE + 17] * u_stride];
            block_hy17[lane] = hy[ev[lane * N_SHAPE + 17] * h_stride];
            block_uz17[lane] = uz[ev[lane * N_SHAPE + 17] * u_stride];
            block_hz17[lane] = hz[ev[lane * N_SHAPE + 17] * h_stride];
            block_ux18[lane] = ux[ev[lane * N_SHAPE + 18] * u_stride];
            block_hx18[lane] = hx[ev[lane * N_SHAPE + 18] * h_stride];
            block_uy18[lane] = uy[ev[lane * N_SHAPE + 18] * u_stride];
            block_hy18[lane] = hy[ev[lane * N_SHAPE + 18] * h_stride];
            block_uz18[lane] = uz[ev[lane * N_SHAPE + 18] * u_stride];
            block_hz18[lane] = hz[ev[lane * N_SHAPE + 18] * h_stride];
            block_ux19[lane] = ux[ev[lane * N_SHAPE + 19] * u_stride];
            block_hx19[lane] = hx[ev[lane * N_SHAPE + 19] * h_stride];
            block_uy19[lane] = uy[ev[lane * N_SHAPE + 19] * u_stride];
            block_hy19[lane] = hy[ev[lane * N_SHAPE + 19] * h_stride];
            block_uz19[lane] = uz[ev[lane * N_SHAPE + 19] * u_stride];
            block_hz19[lane] = hz[ev[lane * N_SHAPE + 19] * h_stride];
            block_ux20[lane] = ux[ev[lane * N_SHAPE + 20] * u_stride];
            block_hx20[lane] = hx[ev[lane * N_SHAPE + 20] * h_stride];
            block_uy20[lane] = uy[ev[lane * N_SHAPE + 20] * u_stride];
            block_hy20[lane] = hy[ev[lane * N_SHAPE + 20] * h_stride];
            block_uz20[lane] = uz[ev[lane * N_SHAPE + 20] * u_stride];
            block_hz20[lane] = hz[ev[lane * N_SHAPE + 20] * h_stride];
            block_ux21[lane] = ux[ev[lane * N_SHAPE + 21] * u_stride];
            block_hx21[lane] = hx[ev[lane * N_SHAPE + 21] * h_stride];
            block_uy21[lane] = uy[ev[lane * N_SHAPE + 21] * u_stride];
            block_hy21[lane] = hy[ev[lane * N_SHAPE + 21] * h_stride];
            block_uz21[lane] = uz[ev[lane * N_SHAPE + 21] * u_stride];
            block_hz21[lane] = hz[ev[lane * N_SHAPE + 21] * h_stride];
            block_ux22[lane] = ux[ev[lane * N_SHAPE + 22] * u_stride];
            block_hx22[lane] = hx[ev[lane * N_SHAPE + 22] * h_stride];
            block_uy22[lane] = uy[ev[lane * N_SHAPE + 22] * u_stride];
            block_hy22[lane] = hy[ev[lane * N_SHAPE + 22] * h_stride];
            block_uz22[lane] = uz[ev[lane * N_SHAPE + 22] * u_stride];
            block_hz22[lane] = hz[ev[lane * N_SHAPE + 22] * h_stride];
            block_ux23[lane] = ux[ev[lane * N_SHAPE + 23] * u_stride];
            block_hx23[lane] = hx[ev[lane * N_SHAPE + 23] * h_stride];
            block_uy23[lane] = uy[ev[lane * N_SHAPE + 23] * u_stride];
            block_hy23[lane] = hy[ev[lane * N_SHAPE + 23] * h_stride];
            block_uz23[lane] = uz[ev[lane * N_SHAPE + 23] * u_stride];
            block_hz23[lane] = hz[ev[lane * N_SHAPE + 23] * h_stride];
            block_ux24[lane] = ux[ev[lane * N_SHAPE + 24] * u_stride];
            block_hx24[lane] = hx[ev[lane * N_SHAPE + 24] * h_stride];
            block_uy24[lane] = uy[ev[lane * N_SHAPE + 24] * u_stride];
            block_hy24[lane] = hy[ev[lane * N_SHAPE + 24] * h_stride];
            block_uz24[lane] = uz[ev[lane * N_SHAPE + 24] * u_stride];
            block_hz24[lane] = hz[ev[lane * N_SHAPE + 24] * h_stride];
            block_ux25[lane] = ux[ev[lane * N_SHAPE + 25] * u_stride];
            block_hx25[lane] = hx[ev[lane * N_SHAPE + 25] * h_stride];
            block_uy25[lane] = uy[ev[lane * N_SHAPE + 25] * u_stride];
            block_hy25[lane] = hy[ev[lane * N_SHAPE + 25] * h_stride];
            block_uz25[lane] = uz[ev[lane * N_SHAPE + 25] * u_stride];
            block_hz25[lane] = hz[ev[lane * N_SHAPE + 25] * h_stride];
            block_ux26[lane] = ux[ev[lane * N_SHAPE + 26] * u_stride];
            block_hx26[lane] = hx[ev[lane * N_SHAPE + 26] * h_stride];
            block_uy26[lane] = uy[ev[lane * N_SHAPE + 26] * u_stride];
            block_hy26[lane] = hy[ev[lane * N_SHAPE + 26] * h_stride];
            block_uz26[lane] = uz[ev[lane * N_SHAPE + 26] * u_stride];
            block_hz26[lane] = hz[ev[lane * N_SHAPE + 26] * h_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outz0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outz1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outz2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outz3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outz4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
            block_outz5[lane] = scalar_t(0);
            block_outx6[lane] = scalar_t(0);
            block_outy6[lane] = scalar_t(0);
            block_outz6[lane] = scalar_t(0);
            block_outx7[lane] = scalar_t(0);
            block_outy7[lane] = scalar_t(0);
            block_outz7[lane] = scalar_t(0);
            block_outx8[lane] = scalar_t(0);
            block_outy8[lane] = scalar_t(0);
            block_outz8[lane] = scalar_t(0);
            block_outx9[lane] = scalar_t(0);
            block_outy9[lane] = scalar_t(0);
            block_outz9[lane] = scalar_t(0);
            block_outx10[lane] = scalar_t(0);
            block_outy10[lane] = scalar_t(0);
            block_outz10[lane] = scalar_t(0);
            block_outx11[lane] = scalar_t(0);
            block_outy11[lane] = scalar_t(0);
            block_outz11[lane] = scalar_t(0);
            block_outx12[lane] = scalar_t(0);
            block_outy12[lane] = scalar_t(0);
            block_outz12[lane] = scalar_t(0);
            block_outx13[lane] = scalar_t(0);
            block_outy13[lane] = scalar_t(0);
            block_outz13[lane] = scalar_t(0);
            block_outx14[lane] = scalar_t(0);
            block_outy14[lane] = scalar_t(0);
            block_outz14[lane] = scalar_t(0);
            block_outx15[lane] = scalar_t(0);
            block_outy15[lane] = scalar_t(0);
            block_outz15[lane] = scalar_t(0);
            block_outx16[lane] = scalar_t(0);
            block_outy16[lane] = scalar_t(0);
            block_outz16[lane] = scalar_t(0);
            block_outx17[lane] = scalar_t(0);
            block_outy17[lane] = scalar_t(0);
            block_outz17[lane] = scalar_t(0);
            block_outx18[lane] = scalar_t(0);
            block_outy18[lane] = scalar_t(0);
            block_outz18[lane] = scalar_t(0);
            block_outx19[lane] = scalar_t(0);
            block_outy19[lane] = scalar_t(0);
            block_outz19[lane] = scalar_t(0);
            block_outx20[lane] = scalar_t(0);
            block_outy20[lane] = scalar_t(0);
            block_outz20[lane] = scalar_t(0);
            block_outx21[lane] = scalar_t(0);
            block_outy21[lane] = scalar_t(0);
            block_outz21[lane] = scalar_t(0);
            block_outx22[lane] = scalar_t(0);
            block_outy22[lane] = scalar_t(0);
            block_outz22[lane] = scalar_t(0);
            block_outx23[lane] = scalar_t(0);
            block_outy23[lane] = scalar_t(0);
            block_outz23[lane] = scalar_t(0);
            block_outx24[lane] = scalar_t(0);
            block_outy24[lane] = scalar_t(0);
            block_outz24[lane] = scalar_t(0);
            block_outx25[lane] = scalar_t(0);
            block_outy25[lane] = scalar_t(0);
            block_outz25[lane] = scalar_t(0);
            block_outx26[lane] = scalar_t(0);
            block_outy26[lane] = scalar_t(0);
            block_outz26[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux8, block_uy8, block_uz8, block_ux1, block_uy1, block_uz1, block_ux11, block_uy11, block_uz11, block_ux24, block_uy24, block_uz24, block_ux9, block_uy9, block_uz9, block_ux3, block_uy3, block_uz3, block_ux10, block_uy10, block_uz10, block_ux2, block_uy2, block_uz2, block_ux16, block_uy16, block_uz16, block_ux20, block_uy20, block_uz20, block_ux17, block_uy17, block_uz17, block_ux23, block_uy23, block_uz23, block_ux26, block_uy26, block_uz26, block_ux21, block_uy21, block_uz21, block_ux19, block_uy19, block_uz19, block_ux22, block_uy22, block_uz22, block_ux18, block_uy18, block_uz18, block_ux4, block_uy4, block_uz4, block_ux12, block_uy12, block_uz12, block_ux5, block_uy5, block_uz5, block_ux15, block_uy15, block_uz15, block_ux25, block_uy25, block_uz25, block_ux13, block_uy13, block_uz13, block_ux7, block_uy7, block_uz7, block_ux14, block_uy14, block_uz14, block_ux6, block_uy6, block_uz6};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx8, block_hy8, block_hz8, block_hx1, block_hy1, block_hz1, block_hx11, block_hy11, block_hz11, block_hx24, block_hy24, block_hz24, block_hx9, block_hy9, block_hz9, block_hx3, block_hy3, block_hz3, block_hx10, block_hy10, block_hz10, block_hx2, block_hy2, block_hz2, block_hx16, block_hy16, block_hz16, block_hx20, block_hy20, block_hz20, block_hx17, block_hy17, block_hz17, block_hx23, block_hy23, block_hz23, block_hx26, block_hy26, block_hz26, block_hx21, block_hy21, block_hz21, block_hx19, block_hy19, block_hz19, block_hx22, block_hy22, block_hz22, block_hx18, block_hy18, block_hz18, block_hx4, block_hy4, block_hz4, block_hx12, block_hy12, block_hz12, block_hx5, block_hy5, block_hz5, block_hx15, block_hy15, block_hz15, block_hx25, block_hy25, block_hz25, block_hx13, block_hy13, block_hz13, block_hx7, block_hy7, block_hz7, block_hx14, block_hy14, block_hz14, block_hx6, block_hy6, block_hz6};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx8, block_outy8, block_outz8, block_outx1, block_outy1, block_outz1, block_outx11, block_outy11, block_outz11, block_outx24, block_outy24, block_outz24, block_outx9, block_outy9, block_outz9, block_outx3, block_outy3, block_outz3, block_outx10, block_outy10, block_outz10, block_outx2, block_outy2, block_outz2, block_outx16, block_outy16, block_outz16, block_outx20, block_outy20, block_outz20, block_outx17, block_outy17, block_outz17, block_outx23, block_outy23, block_outz23, block_outx26, block_outy26, block_outz26, block_outx21, block_outy21, block_outz21, block_outx19, block_outy19, block_outz19, block_outx22, block_outy22, block_outz22, block_outx18, block_outy18, block_outz18, block_outx4, block_outy4, block_outz4, block_outx12, block_outy12, block_outz12, block_outx5, block_outy5, block_outz5, block_outx15, block_outy15, block_outz15, block_outx25, block_outy25, block_outz25, block_outx13, block_outy13, block_outz13, block_outx7, block_outy7, block_outz7, block_outx14, block_outy14, block_outz14, block_outx6, block_outy6, block_outz6};

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_x0, block_y0, block_z0, block_x8, block_y8, block_z8, block_x1, block_y1, block_z1, block_x11, block_y11, block_z11, block_x24, block_y24, block_z24, block_x9, block_y9, block_z9, block_x3, block_y3, block_z3, block_x10, block_y10, block_z10, block_x2, block_y2, block_z2, block_x16, block_y16, block_z16, block_x20, block_y20, block_z20, block_x17, block_y17, block_z17, block_x23, block_y23, block_z23, block_x26, block_y26, block_z26, block_x21, block_y21, block_z21, block_x19, block_y19, block_z19, block_x22, block_y22, block_z22, block_x18, block_y18, block_z18, block_x4, block_y4, block_z4, block_x12, block_y12, block_z12, block_x5, block_y5, block_z5, block_x15, block_y15, block_z15, block_x25, block_y25, block_z25, block_x13, block_y13, block_z13, block_x7, block_y7, block_z7, block_x14, block_y14, block_z14, block_x6, block_y6, block_z6};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_streams, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11 * J22 - J12 * J21;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = J02 * J21 - J01 * J22;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = J01 * J12 - J02 * J11;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J12 * J20 - J10 * J22;
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = J00 * J22 - J02 * J20;
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = J02 * J10 - J00 * J12;
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = J10 * J21 - J11 * J20;
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = J01 * J20 - J00 * J21;
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
            }
        }

        generated_neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, shape_1d, grad_1d, q_weight_1d, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 0] * out_stride] += block_outz0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 1] * out_stride] += block_outz1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 2] * out_stride] += block_outz2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 3] * out_stride] += block_outz3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 4] * out_stride] += block_outz4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 5] * out_stride] += block_outz5[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 6] * out_stride] += block_outx6[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 6] * out_stride] += block_outy6[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 6] * out_stride] += block_outz6[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 7] * out_stride] += block_outx7[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 7] * out_stride] += block_outy7[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 7] * out_stride] += block_outz7[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 8] * out_stride] += block_outx8[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 8] * out_stride] += block_outy8[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 8] * out_stride] += block_outz8[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 9] * out_stride] += block_outx9[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 9] * out_stride] += block_outy9[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 9] * out_stride] += block_outz9[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 10] * out_stride] += block_outx10[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 10] * out_stride] += block_outy10[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 10] * out_stride] += block_outz10[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 11] * out_stride] += block_outx11[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 11] * out_stride] += block_outy11[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 11] * out_stride] += block_outz11[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 12] * out_stride] += block_outx12[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 12] * out_stride] += block_outy12[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 12] * out_stride] += block_outz12[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 13] * out_stride] += block_outx13[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 13] * out_stride] += block_outy13[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 13] * out_stride] += block_outz13[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 14] * out_stride] += block_outx14[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 14] * out_stride] += block_outy14[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 14] * out_stride] += block_outz14[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 15] * out_stride] += block_outx15[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 15] * out_stride] += block_outy15[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 15] * out_stride] += block_outz15[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 16] * out_stride] += block_outx16[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 16] * out_stride] += block_outy16[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 16] * out_stride] += block_outz16[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 17] * out_stride] += block_outx17[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 17] * out_stride] += block_outy17[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 17] * out_stride] += block_outz17[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 18] * out_stride] += block_outx18[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 18] * out_stride] += block_outy18[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 18] * out_stride] += block_outz18[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 19] * out_stride] += block_outx19[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 19] * out_stride] += block_outy19[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 19] * out_stride] += block_outz19[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 20] * out_stride] += block_outx20[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 20] * out_stride] += block_outy20[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 20] * out_stride] += block_outz20[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 21] * out_stride] += block_outx21[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 21] * out_stride] += block_outy21[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 21] * out_stride] += block_outz21[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 22] * out_stride] += block_outx22[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 22] * out_stride] += block_outy22[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 22] * out_stride] += block_outz22[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 23] * out_stride] += block_outx23[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 23] * out_stride] += block_outy23[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 23] * out_stride] += block_outz23[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 24] * out_stride] += block_outx24[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 24] * out_stride] += block_outy24[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 24] * out_stride] += block_outz24[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 25] * out_stride] += block_outx25[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 25] * out_stride] += block_outy25[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 25] * out_stride] += block_outz25[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 26] * out_stride] += block_outx26[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 26] * out_stride] += block_outy26[lane];

#pragma omp atomic update
            outz[ev[lane * N_SHAPE + 26] * out_stride] += block_outz26[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

