#include "generated_neohookean_ogden_d3_simplex_local.hpp"
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

static const real_t generated_neohookean_ogden_tet10_tet10_q_weight[4] = {real_t(0.041666666666666664), real_t(0.041666666666666664), real_t(0.041666666666666664), real_t(0.041666666666666664)};
static const real_t generated_neohookean_ogden_tet10_tet10_grad_ref_x[40] = {real_t(-1.3416407864998741), real_t(-0.44721359549995798), real_t(0), real_t(0), real_t(1.7888543819998319), real_t(0.55278640450004202), real_t(-0.55278640450004202), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(0), real_t(0.44721359549995832), real_t(1.3416407864998741), real_t(0), real_t(0), real_t(-1.7888543819998315), real_t(0.55278640450004202), real_t(-0.55278640450004202), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(0), real_t(0.44721359549995832), real_t(-0.44721359549995798), real_t(0), real_t(0), real_t(0), real_t(2.3416407864998741), real_t(-2.3416407864998741), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(0), real_t(0.44721359549995832), real_t(-0.44721359549995798), real_t(0), real_t(0), real_t(0), real_t(0.55278640450004202), real_t(-0.55278640450004202), real_t(-2.3416407864998741), real_t(2.3416407864998741), real_t(0)};
static const real_t generated_neohookean_ogden_tet10_tet10_grad_ref_y[40] = {real_t(-1.3416407864998741), real_t(0), real_t(-0.44721359549995798), real_t(0), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(1.7888543819998319), real_t(-0.55278640450004202), real_t(0), real_t(0.55278640450004202), real_t(0.44721359549995832), real_t(0), real_t(-0.44721359549995798), real_t(0), real_t(-2.3416407864998741), real_t(2.3416407864998741), real_t(0), real_t(-0.55278640450004202), real_t(0), real_t(0.55278640450004202), real_t(0.44721359549995832), real_t(0), real_t(1.3416407864998741), real_t(0), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(-1.7888543819998315), real_t(-0.55278640450004202), real_t(0), real_t(0.55278640450004202), real_t(0.44721359549995832), real_t(0), real_t(-0.44721359549995798), real_t(0), real_t(-0.55278640450004202), real_t(0.55278640450004202), real_t(0), real_t(-2.3416407864998741), real_t(0), real_t(2.3416407864998741)};
static const real_t generated_neohookean_ogden_tet10_tet10_grad_ref_z[40] = {real_t(-1.3416407864998741), real_t(0), real_t(0), real_t(-0.44721359549995798), real_t(-0.55278640450004202), real_t(0), real_t(-0.55278640450004202), real_t(1.7888543819998319), real_t(0.55278640450004202), real_t(0.55278640450004202), real_t(0.44721359549995832), real_t(0), real_t(0), real_t(-0.44721359549995798), real_t(-2.3416407864998741), real_t(0), real_t(-0.55278640450004202), real_t(0), real_t(2.3416407864998741), real_t(0.55278640450004202), real_t(0.44721359549995832), real_t(0), real_t(0), real_t(-0.44721359549995798), real_t(-0.55278640450004202), real_t(0), real_t(-2.3416407864998741), real_t(0), real_t(0.55278640450004202), real_t(2.3416407864998741), real_t(0.44721359549995832), real_t(0), real_t(0), real_t(1.3416407864998741), real_t(-0.55278640450004202), real_t(0), real_t(-0.55278640450004202), real_t(-1.7888543819998315), real_t(0.55278640450004202), real_t(0.55278640450004202)};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet10_tet10_objective_soa",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    120,
    4,
    2,
    30,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet10_tet10_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_objective_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};

        generated_neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_soa(
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
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_soa_impl<real_t, 4, 10, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_isoparametric_soa_impl<real_t, 4, 10, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};

        generated_neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet10_tet10_gradient_soa",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    120,
    4,
    2,
    30,
    0,
    30,
    30,
    30,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet10_tet10_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_gradient_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT outz9
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        generated_neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_out_streams);

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_soa(
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
        real_t *const SFEM_RESTRICT outz9
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_soa_impl<real_t, 4, 10, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT outz9
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_out_streams);

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT outz9
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_soa_impl<real_t, 4, 10, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        generated_neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_out_streams);

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_out_streams);

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet10_tet10_apply_soa",
    "TET10",
    3,
    4,
    10,
    16,
    2,
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
    120,
    4,
    2,
    30,
    30,
    30,
    30,
    30,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet10_tet10_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_apply_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT outz9
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3, block_hx4, block_hy4, block_hz4, block_hx5, block_hy5, block_hz5, block_hx6, block_hy6, block_hz6, block_hx7, block_hy7, block_hz7, block_hx8, block_hy8, block_hz8, block_hx9, block_hy9, block_hz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        generated_neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_adjugate4 + evbegin, jacobian_adjugate5 + evbegin, jacobian_adjugate6 + evbegin, jacobian_adjugate7 + evbegin, jacobian_adjugate8 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_soa(
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
        real_t *const SFEM_RESTRICT outz9
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_soa_impl<real_t, 4, 10, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, hx4, hy4, hz4, hx5, hy5, hz5, hx6, hy6, hz6, hx7, hy7, hz7, hx8, hy8, hz8, hx9, hy9, hz9, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_soa_impl(
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
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        real_t *const SFEM_RESTRICT outz9
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 4, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 10, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3, block_hx4, block_hy4, block_hz4, block_hx5, block_hy5, block_hz5, block_hx6, block_hy6, block_hz6, block_hx7, block_hy7, block_hz7, block_hx8, block_hy8, block_hz8, block_hx9, block_hy9, block_hz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT outz9
) {
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_isoparametric_soa_impl<real_t, 4, 10, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, x9, y9, z9, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet10_tet10_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet10_tet10_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, ux4, uy4, uz4, ux5, uy5, uz5, ux6, uy6, uz6, ux7, uy7, uz7, ux8, uy8, uz8, ux9, uy9, uz9, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, hx4, hy4, hz4, hx5, hy5, hz5, hx6, hy6, hz6, hx7, hy7, hz7, hx8, hy8, hz8, hx9, hy9, hz9, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3, outx4, outy4, outz4, outx5, outy5, outz5, outx6, outy6, outz6, outx7, outy7, outz7, outx8, outy8, outz8, outx9, outy9, outz9);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3, block_hx4, block_hy4, block_hz4, block_hx5, block_hy5, block_hz5, block_hx6, block_hy6, block_hz6, block_hx7, block_hy7, block_hz7, block_hx8, block_hy8, block_hz8, block_hx9, block_hy9, block_hz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        generated_neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
    static const scalar_t grad_ref_y[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
    static const scalar_t grad_ref_z[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
    static const scalar_t q_weight[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};

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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3, block_ux4, block_uy4, block_uz4, block_ux5, block_uy5, block_uz5, block_ux6, block_uy6, block_uz6, block_ux7, block_uy7, block_uz7, block_ux8, block_uy8, block_uz8, block_ux9, block_uy9, block_uz9};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3, block_hx4, block_hy4, block_hz4, block_hx5, block_hy5, block_hz5, block_hx6, block_hy6, block_hz6, block_hx7, block_hy7, block_hz7, block_hx8, block_hy8, block_hz8, block_hx9, block_hy9, block_hz9};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3, block_outx4, block_outy4, block_outz4, block_outx5, block_outy5, block_outz5, block_outx6, block_outy6, block_outz6, block_outx7, block_outy7, block_outz7, block_outx8, block_outy8, block_outz8, block_outx9, block_outy9, block_outz9};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3, block_x4, block_y4, block_z4, block_x5, block_y5, block_z5, block_x6, block_y6, block_z6, block_x7, block_y7, block_z7, block_x8, block_y8, block_z8, block_x9, block_y9, block_z9};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J02 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                scalar_t J12 = scalar_t(0);
                scalar_t J20 = scalar_t(0);
                scalar_t J21 = scalar_t(0);
                scalar_t J22 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 3 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 3 + 0][lane] * g1;
                    J02 += block_coordinate_streams[shape * 3 + 0][lane] * g2;
                    J10 += block_coordinate_streams[shape * 3 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 3 + 1][lane] * g1;
                    J12 += block_coordinate_streams[shape * 3 + 1][lane] * g2;
                    J20 += block_coordinate_streams[shape * 3 + 2][lane] * g0;
                    J21 += block_coordinate_streams[shape * 3 + 2][lane] * g1;
                    J22 += block_coordinate_streams[shape * 3 + 2][lane] * g2;
                }
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

        generated_neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, grad_ref_x, grad_ref_y, grad_ref_z, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

