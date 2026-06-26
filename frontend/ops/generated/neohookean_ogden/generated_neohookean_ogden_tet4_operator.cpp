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

static const real_t generated_neohookean_ogden_tet4_tet4_q_weight[1] = {real_t(0.16666666666666666)};
static const real_t generated_neohookean_ogden_tet4_tet4_grad_ref_x[4] = {real_t(-1), real_t(1), real_t(0), real_t(0)};
static const real_t generated_neohookean_ogden_tet4_tet4_grad_ref_y[4] = {real_t(-1), real_t(0), real_t(1), real_t(0)};
static const real_t generated_neohookean_ogden_tet4_tet4_grad_ref_z[4] = {real_t(-1), real_t(0), real_t(0), real_t(1)};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet4_tet4_objective_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
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
    12,
    1,
    2,
    12,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet4_tet4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_objective_soa_impl(
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
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};

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

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_soa(
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
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_soa_impl<real_t, 1, 4, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_soa_impl(
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
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_isoparametric_soa_impl<real_t, 1, 4, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};

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

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet4_tet4_gradient_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
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
    12,
    1,
    2,
    12,
    0,
    12,
    12,
    12,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet4_tet4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_gradient_soa_impl(
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
        real_t *const SFEM_RESTRICT outz3
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_soa(
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
        real_t *const SFEM_RESTRICT outz3
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_soa_impl<real_t, 1, 4, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_soa_impl(
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
        real_t *const SFEM_RESTRICT outz3
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT outz3
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_soa_impl<real_t, 1, 4, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data = {
    "generated_neohookean_ogden_tet4_tet4_apply_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
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
    12,
    1,
    2,
    12,
    12,
    12,
    12,
    12,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tet4_tet4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_apply_soa_impl(
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
        real_t *const SFEM_RESTRICT outz3
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_soa(
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
        real_t *const SFEM_RESTRICT outz3
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_soa_impl<real_t, 1, 4, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_adjugate4, jacobian_adjugate5, jacobian_adjugate6, jacobian_adjugate7, jacobian_adjugate8, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_soa_impl(
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
        real_t *const SFEM_RESTRICT outz3
) {
    static constexpr int DIM = 3;
    static_assert(N_QP == 1, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 4, "N_SHAPE does not match generated expression");
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_soa(
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
        real_t *const SFEM_RESTRICT outz3
) {
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_isoparametric_soa_impl<real_t, 1, 4, 16>(nelements, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tet4_tet4_grad_ref_z, sfem::codegen::generated_neohookean_ogden_tet4_tet4_q_weight, mu, lmbda, ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3, hx0, hy0, hz0, hx1, hy1, hz1, hx2, hy2, hz2, hx3, hy3, hz3, outx0, outy0, outz0, outx1, outy1, outz1, outx2, outy2, outz2, outx3, outy3, outz3);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    static const scalar_t grad_ref_x[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
    static const scalar_t grad_ref_y[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
    static const scalar_t grad_ref_z[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
    static const scalar_t q_weight[1] = {scalar_t(0.16666666666666666)};

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

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
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
        }

        const scalar_t *const block_u_streams[N_SHAPE * 3] = {block_ux0, block_uy0, block_uz0, block_ux1, block_uy1, block_uz1, block_ux2, block_uy2, block_uz2, block_ux3, block_uy3, block_uz3};
        const scalar_t *const block_h_streams[N_SHAPE * 3] = {block_hx0, block_hy0, block_hz0, block_hx1, block_hy1, block_hz1, block_hx2, block_hy2, block_hz2, block_hx3, block_hy3, block_hz3};
        scalar_t *const block_out_streams[N_SHAPE * 3] = {block_outx0, block_outy0, block_outz0, block_outx1, block_outy1, block_outz1, block_outx2, block_outy2, block_outz2, block_outx3, block_outy3, block_outz3};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 3] = {block_x0, block_y0, block_z0, block_x1, block_y1, block_z1, block_x2, block_y2, block_z2, block_x3, block_y3, block_z3};

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

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_float(
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
    return sfem::codegen::generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

