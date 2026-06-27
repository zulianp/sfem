#include "generated_neohookean_ogden_d2_simplex_local.hpp"
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

static const real_t generated_neohookean_ogden_tri6_tri6_q_weight[6] = {real_t(0.11169079483900569), real_t(0.11169079483900569), real_t(0.11169079483900569), real_t(0.054975871827660998), real_t(0.054975871827660998), real_t(0.054975871827660998)};
static const real_t generated_neohookean_ogden_tri6_tri6_grad_ref_x[36] = {real_t(0.56758792732771912), real_t(0.78379396366385956), real_t(0), real_t(-1.3513818909915787), real_t(1.7837939636638596), real_t(-1.7837939636638596), real_t(-0.78379396366385956), real_t(-0.56758792732771912), real_t(0), real_t(1.3513818909915787), real_t(1.7837939636638596), real_t(-1.7837939636638596), real_t(-0.78379396366385956), real_t(0.78379396366385956), real_t(0), real_t(5.5511151231257827e-17), real_t(0.43241207267228082), real_t(-0.43241207267228082), real_t(-2.2673902919218341), real_t(-0.63369514596091703), real_t(0), real_t(2.9010854378827511), real_t(0.36630485403908297), real_t(-0.36630485403908297), real_t(0.63369514596091703), real_t(2.2673902919218341), real_t(0), real_t(-2.9010854378827511), real_t(0.36630485403908297), real_t(-0.36630485403908297), real_t(0.63369514596091703), real_t(-0.63369514596091703), real_t(0), real_t(0), real_t(3.2673902919218341), real_t(-3.2673902919218341)};
static const real_t generated_neohookean_ogden_tri6_tri6_grad_ref_y[36] = {real_t(0.56758792732771912), real_t(0), real_t(0.78379396366385956), real_t(-1.7837939636638596), real_t(1.7837939636638596), real_t(-1.3513818909915787), real_t(-0.78379396366385956), real_t(0), real_t(0.78379396366385956), real_t(-0.43241207267228082), real_t(0.43241207267228082), real_t(0), real_t(-0.78379396366385956), real_t(0), real_t(-0.56758792732771912), real_t(-1.7837939636638596), real_t(1.7837939636638596), real_t(1.3513818909915787), real_t(-2.2673902919218341), real_t(0), real_t(-0.63369514596091703), real_t(-0.36630485403908297), real_t(0.36630485403908297), real_t(2.9010854378827511), real_t(0.63369514596091703), real_t(0), real_t(-0.63369514596091703), real_t(-3.2673902919218341), real_t(3.2673902919218341), real_t(0), real_t(0.63369514596091703), real_t(0), real_t(2.2673902919218341), real_t(-0.36630485403908297), real_t(0.36630485403908297), real_t(-2.9010854378827511)};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data = {
    "generated_neohookean_ogden_tri6_tri6_objective_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    15,
    27,
    1,
    0,
    5,
    0,
    1,
    0,
    4,
    13,
    75,
    8,
    13,
    5,
    72,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tri6_tri6_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_objective_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};

        generated_neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_soa_impl<real_t, 6, 6, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_value[lane] = value[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] = block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_isoparametric_soa_impl<real_t, 6, 6, 16>(nelements, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, value);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};

        generated_neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_value);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data = {
    "generated_neohookean_ogden_tri6_tri6_gradient_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    15,
    27,
    1,
    0,
    5,
    0,
    1,
    0,
    4,
    13,
    75,
    8,
    13,
    5,
    72,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tri6_tri6_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_gradient_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        generated_neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_soa_impl<real_t, 6, 6, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, outx0, outy0, outx1, outy1, outx2, outy2, outx3, outy3, outx4, outy4, outx5, outy5);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_soa_impl<real_t, 6, 6, 16>(nelements, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, outx0, outy0, outx1, outy1, outx2, outy2, outx3, outy3, outx4, outy4, outx5, outy5);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        generated_neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data = {
    "generated_neohookean_ogden_tri6_tri6_apply_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    48,
    95,
    1,
    0,
    6,
    0,
    1,
    0,
    4,
    40,
    177,
    31,
    26,
    5,
    72,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics(void) {
    return &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data;
}

extern "C" double generated_neohookean_ogden_tri6_tri6_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_apply_soa_impl(
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_hx0[lane] = hx0[evbegin + lane];
            block_hy0[lane] = hy0[evbegin + lane];
            block_hx1[lane] = hx1[evbegin + lane];
            block_hy1[lane] = hy1[evbegin + lane];
            block_hx2[lane] = hx2[evbegin + lane];
            block_hy2[lane] = hy2[evbegin + lane];
            block_hx3[lane] = hx3[evbegin + lane];
            block_hy3[lane] = hy3[evbegin + lane];
            block_hx4[lane] = hx4[evbegin + lane];
            block_hy4[lane] = hy4[evbegin + lane];
            block_hx5[lane] = hx5[evbegin + lane];
            block_hy5[lane] = hy5[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        const scalar_t *const block_h_streams[N_SHAPE * 2] = {block_hx0, block_hy0, block_hx1, block_hy1, block_hx2, block_hy2, block_hx3, block_hy3, block_hx4, block_hy4, block_hx5, block_hy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        generated_neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, nelements, jacobian_adjugate0 + evbegin, jacobian_adjugate1 + evbegin, jacobian_adjugate2 + evbegin, jacobian_adjugate3 + evbegin, jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT jacobian_adjugate0,
        const real_t *const SFEM_RESTRICT jacobian_adjugate1,
        const real_t *const SFEM_RESTRICT jacobian_adjugate2,
        const real_t *const SFEM_RESTRICT jacobian_adjugate3,
        const real_t *const SFEM_RESTRICT jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_soa_impl<real_t, 6, 6, 16>(nelements, jacobian_adjugate0, jacobian_adjugate1, jacobian_adjugate2, jacobian_adjugate3, jacobian_determinant0, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, hx0, hy0, hx1, hy1, hx2, hy2, hx3, hy3, hx4, hy4, hx5, hy5, outx0, outy0, outx1, outy1, outx2, outy2, outx3, outy3, outx4, outy4, outx5, outy5);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_soa_impl(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    static constexpr int DIM = 2;
    static_assert(N_QP == 6, "N_QP does not match generated geometry streams");
    static_assert(N_SHAPE == 6, "N_SHAPE does not match generated expression");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x0[evbegin + lane];
            block_y0[lane] = y0[evbegin + lane];
            block_x1[lane] = x1[evbegin + lane];
            block_y1[lane] = y1[evbegin + lane];
            block_x2[lane] = x2[evbegin + lane];
            block_y2[lane] = y2[evbegin + lane];
            block_x3[lane] = x3[evbegin + lane];
            block_y3[lane] = y3[evbegin + lane];
            block_x4[lane] = x4[evbegin + lane];
            block_y4[lane] = y4[evbegin + lane];
            block_x5[lane] = x5[evbegin + lane];
            block_y5[lane] = y5[evbegin + lane];
            block_ux0[lane] = ux0[evbegin + lane];
            block_uy0[lane] = uy0[evbegin + lane];
            block_ux1[lane] = ux1[evbegin + lane];
            block_uy1[lane] = uy1[evbegin + lane];
            block_ux2[lane] = ux2[evbegin + lane];
            block_uy2[lane] = uy2[evbegin + lane];
            block_ux3[lane] = ux3[evbegin + lane];
            block_uy3[lane] = uy3[evbegin + lane];
            block_ux4[lane] = ux4[evbegin + lane];
            block_uy4[lane] = uy4[evbegin + lane];
            block_ux5[lane] = ux5[evbegin + lane];
            block_uy5[lane] = uy5[evbegin + lane];
            block_hx0[lane] = hx0[evbegin + lane];
            block_hy0[lane] = hy0[evbegin + lane];
            block_hx1[lane] = hx1[evbegin + lane];
            block_hy1[lane] = hy1[evbegin + lane];
            block_hx2[lane] = hx2[evbegin + lane];
            block_hy2[lane] = hy2[evbegin + lane];
            block_hx3[lane] = hx3[evbegin + lane];
            block_hy3[lane] = hy3[evbegin + lane];
            block_hx4[lane] = hx4[evbegin + lane];
            block_hy4[lane] = hy4[evbegin + lane];
            block_hx5[lane] = hx5[evbegin + lane];
            block_hy5[lane] = hy5[evbegin + lane];
            block_outx0[lane] = outx0[evbegin + lane];
            block_outy0[lane] = outy0[evbegin + lane];
            block_outx1[lane] = outx1[evbegin + lane];
            block_outy1[lane] = outy1[evbegin + lane];
            block_outx2[lane] = outx2[evbegin + lane];
            block_outy2[lane] = outy2[evbegin + lane];
            block_outx3[lane] = outx3[evbegin + lane];
            block_outy3[lane] = outy3[evbegin + lane];
            block_outx4[lane] = outx4[evbegin + lane];
            block_outy4[lane] = outy4[evbegin + lane];
            block_outx5[lane] = outx5[evbegin + lane];
            block_outy5[lane] = outy5[evbegin + lane];
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        const scalar_t *const block_h_streams[N_SHAPE * 2] = {block_hx0, block_hy0, block_hx1, block_hy1, block_hx2, block_hy2, block_hx3, block_hy3, block_hx4, block_hy4, block_hx5, block_hy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            outx0[evbegin + lane] = block_outx0[lane];
            outy0[evbegin + lane] = block_outy0[lane];
            outx1[evbegin + lane] = block_outx1[lane];
            outy1[evbegin + lane] = block_outy1[lane];
            outx2[evbegin + lane] = block_outx2[lane];
            outy2[evbegin + lane] = block_outy2[lane];
            outx3[evbegin + lane] = block_outx3[lane];
            outy3[evbegin + lane] = block_outy3[lane];
            outx4[evbegin + lane] = block_outx4[lane];
            outy4[evbegin + lane] = block_outy4[lane];
            outx5[evbegin + lane] = block_outx5[lane];
            outy5[evbegin + lane] = block_outy5[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_soa(
        const ptrdiff_t nelements,
        const real_t *const SFEM_RESTRICT x0,
        const real_t *const SFEM_RESTRICT y0,
        const real_t *const SFEM_RESTRICT x1,
        const real_t *const SFEM_RESTRICT y1,
        const real_t *const SFEM_RESTRICT x2,
        const real_t *const SFEM_RESTRICT y2,
        const real_t *const SFEM_RESTRICT x3,
        const real_t *const SFEM_RESTRICT y3,
        const real_t *const SFEM_RESTRICT x4,
        const real_t *const SFEM_RESTRICT y4,
        const real_t *const SFEM_RESTRICT x5,
        const real_t *const SFEM_RESTRICT y5,
        const real_t mu,
        const real_t lmbda,
        const real_t *const SFEM_RESTRICT ux0,
        const real_t *const SFEM_RESTRICT uy0,
        const real_t *const SFEM_RESTRICT ux1,
        const real_t *const SFEM_RESTRICT uy1,
        const real_t *const SFEM_RESTRICT ux2,
        const real_t *const SFEM_RESTRICT uy2,
        const real_t *const SFEM_RESTRICT ux3,
        const real_t *const SFEM_RESTRICT uy3,
        const real_t *const SFEM_RESTRICT ux4,
        const real_t *const SFEM_RESTRICT uy4,
        const real_t *const SFEM_RESTRICT ux5,
        const real_t *const SFEM_RESTRICT uy5,
        const real_t *const SFEM_RESTRICT hx0,
        const real_t *const SFEM_RESTRICT hy0,
        const real_t *const SFEM_RESTRICT hx1,
        const real_t *const SFEM_RESTRICT hy1,
        const real_t *const SFEM_RESTRICT hx2,
        const real_t *const SFEM_RESTRICT hy2,
        const real_t *const SFEM_RESTRICT hx3,
        const real_t *const SFEM_RESTRICT hy3,
        const real_t *const SFEM_RESTRICT hx4,
        const real_t *const SFEM_RESTRICT hy4,
        const real_t *const SFEM_RESTRICT hx5,
        const real_t *const SFEM_RESTRICT hy5,
        real_t *const SFEM_RESTRICT outx0,
        real_t *const SFEM_RESTRICT outy0,
        real_t *const SFEM_RESTRICT outx1,
        real_t *const SFEM_RESTRICT outy1,
        real_t *const SFEM_RESTRICT outx2,
        real_t *const SFEM_RESTRICT outy2,
        real_t *const SFEM_RESTRICT outx3,
        real_t *const SFEM_RESTRICT outy3,
        real_t *const SFEM_RESTRICT outx4,
        real_t *const SFEM_RESTRICT outy4,
        real_t *const SFEM_RESTRICT outx5,
        real_t *const SFEM_RESTRICT outy5
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_isoparametric_soa_impl<real_t, 6, 6, 16>(nelements, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_x, sfem::codegen::generated_neohookean_ogden_tri6_tri6_grad_ref_y, sfem::codegen::generated_neohookean_ogden_tri6_tri6_q_weight, mu, lmbda, ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5, hx0, hy0, hx1, hy1, hx2, hy2, hx3, hy3, hx4, hy4, hx5, hy5, outx0, outy0, outx1, outy1, outx2, outy2, outx3, outy3, outx4, outy4, outx5, outy5);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_hx0[lane] = hx[ev[lane * N_SHAPE + 0] * h_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_hy0[lane] = hy[ev[lane * N_SHAPE + 0] * h_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_hx1[lane] = hx[ev[lane * N_SHAPE + 1] * h_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_hy1[lane] = hy[ev[lane * N_SHAPE + 1] * h_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_hx2[lane] = hx[ev[lane * N_SHAPE + 2] * h_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_hy2[lane] = hy[ev[lane * N_SHAPE + 2] * h_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_hx3[lane] = hx[ev[lane * N_SHAPE + 3] * h_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_hy3[lane] = hy[ev[lane * N_SHAPE + 3] * h_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_hx4[lane] = hx[ev[lane * N_SHAPE + 4] * h_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_hy4[lane] = hy[ev[lane * N_SHAPE + 4] * h_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_hx5[lane] = hx[ev[lane * N_SHAPE + 5] * h_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_hy5[lane] = hy[ev[lane * N_SHAPE + 5] * h_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        const scalar_t *const block_h_streams[N_SHAPE * 2] = {block_hx0, block_hy0, block_hx1, block_hy1, block_hx2, block_hy2, block_hx3, block_hy3, block_hx4, block_hy4, block_hx5, block_hy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        generated_neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const double *const SFEM_RESTRICT g_jacobian_adjugate0,
        const double *const SFEM_RESTRICT g_jacobian_adjugate1,
        const double *const SFEM_RESTRICT g_jacobian_adjugate2,
        const double *const SFEM_RESTRICT g_jacobian_adjugate3,
        const double *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl<double>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const float *const SFEM_RESTRICT g_jacobian_adjugate0,
        const float *const SFEM_RESTRICT g_jacobian_adjugate1,
        const float *const SFEM_RESTRICT g_jacobian_adjugate2,
        const float *const SFEM_RESTRICT g_jacobian_adjugate3,
        const float *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl<float>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    static const scalar_t grad_ref_x[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
    static const scalar_t grad_ref_y[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
    static const scalar_t q_weight[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_x0[VECTOR_SIZE];
        scalar_t block_y0[VECTOR_SIZE];
        scalar_t block_x1[VECTOR_SIZE];
        scalar_t block_y1[VECTOR_SIZE];
        scalar_t block_x2[VECTOR_SIZE];
        scalar_t block_y2[VECTOR_SIZE];
        scalar_t block_x3[VECTOR_SIZE];
        scalar_t block_y3[VECTOR_SIZE];
        scalar_t block_x4[VECTOR_SIZE];
        scalar_t block_y4[VECTOR_SIZE];
        scalar_t block_x5[VECTOR_SIZE];
        scalar_t block_y5[VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t block_ux0[VECTOR_SIZE];
        scalar_t block_uy0[VECTOR_SIZE];
        scalar_t block_ux1[VECTOR_SIZE];
        scalar_t block_uy1[VECTOR_SIZE];
        scalar_t block_ux2[VECTOR_SIZE];
        scalar_t block_uy2[VECTOR_SIZE];
        scalar_t block_ux3[VECTOR_SIZE];
        scalar_t block_uy3[VECTOR_SIZE];
        scalar_t block_ux4[VECTOR_SIZE];
        scalar_t block_uy4[VECTOR_SIZE];
        scalar_t block_ux5[VECTOR_SIZE];
        scalar_t block_uy5[VECTOR_SIZE];
        scalar_t block_hx0[VECTOR_SIZE];
        scalar_t block_hy0[VECTOR_SIZE];
        scalar_t block_hx1[VECTOR_SIZE];
        scalar_t block_hy1[VECTOR_SIZE];
        scalar_t block_hx2[VECTOR_SIZE];
        scalar_t block_hy2[VECTOR_SIZE];
        scalar_t block_hx3[VECTOR_SIZE];
        scalar_t block_hy3[VECTOR_SIZE];
        scalar_t block_hx4[VECTOR_SIZE];
        scalar_t block_hy4[VECTOR_SIZE];
        scalar_t block_hx5[VECTOR_SIZE];
        scalar_t block_hy5[VECTOR_SIZE];
        scalar_t block_outx0[VECTOR_SIZE];
        scalar_t block_outy0[VECTOR_SIZE];
        scalar_t block_outx1[VECTOR_SIZE];
        scalar_t block_outy1[VECTOR_SIZE];
        scalar_t block_outx2[VECTOR_SIZE];
        scalar_t block_outy2[VECTOR_SIZE];
        scalar_t block_outx3[VECTOR_SIZE];
        scalar_t block_outy3[VECTOR_SIZE];
        scalar_t block_outx4[VECTOR_SIZE];
        scalar_t block_outy4[VECTOR_SIZE];
        scalar_t block_outx5[VECTOR_SIZE];
        scalar_t block_outy5[VECTOR_SIZE];

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_x0[lane] = x[ev[lane * N_SHAPE + 0]];
            block_y0[lane] = y[ev[lane * N_SHAPE + 0]];
            block_x1[lane] = x[ev[lane * N_SHAPE + 1]];
            block_y1[lane] = y[ev[lane * N_SHAPE + 1]];
            block_x2[lane] = x[ev[lane * N_SHAPE + 2]];
            block_y2[lane] = y[ev[lane * N_SHAPE + 2]];
            block_x3[lane] = x[ev[lane * N_SHAPE + 3]];
            block_y3[lane] = y[ev[lane * N_SHAPE + 3]];
            block_x4[lane] = x[ev[lane * N_SHAPE + 4]];
            block_y4[lane] = y[ev[lane * N_SHAPE + 4]];
            block_x5[lane] = x[ev[lane * N_SHAPE + 5]];
            block_y5[lane] = y[ev[lane * N_SHAPE + 5]];
        }

#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            block_ux0[lane] = ux[ev[lane * N_SHAPE + 0] * u_stride];
            block_hx0[lane] = hx[ev[lane * N_SHAPE + 0] * h_stride];
            block_uy0[lane] = uy[ev[lane * N_SHAPE + 0] * u_stride];
            block_hy0[lane] = hy[ev[lane * N_SHAPE + 0] * h_stride];
            block_ux1[lane] = ux[ev[lane * N_SHAPE + 1] * u_stride];
            block_hx1[lane] = hx[ev[lane * N_SHAPE + 1] * h_stride];
            block_uy1[lane] = uy[ev[lane * N_SHAPE + 1] * u_stride];
            block_hy1[lane] = hy[ev[lane * N_SHAPE + 1] * h_stride];
            block_ux2[lane] = ux[ev[lane * N_SHAPE + 2] * u_stride];
            block_hx2[lane] = hx[ev[lane * N_SHAPE + 2] * h_stride];
            block_uy2[lane] = uy[ev[lane * N_SHAPE + 2] * u_stride];
            block_hy2[lane] = hy[ev[lane * N_SHAPE + 2] * h_stride];
            block_ux3[lane] = ux[ev[lane * N_SHAPE + 3] * u_stride];
            block_hx3[lane] = hx[ev[lane * N_SHAPE + 3] * h_stride];
            block_uy3[lane] = uy[ev[lane * N_SHAPE + 3] * u_stride];
            block_hy3[lane] = hy[ev[lane * N_SHAPE + 3] * h_stride];
            block_ux4[lane] = ux[ev[lane * N_SHAPE + 4] * u_stride];
            block_hx4[lane] = hx[ev[lane * N_SHAPE + 4] * h_stride];
            block_uy4[lane] = uy[ev[lane * N_SHAPE + 4] * u_stride];
            block_hy4[lane] = hy[ev[lane * N_SHAPE + 4] * h_stride];
            block_ux5[lane] = ux[ev[lane * N_SHAPE + 5] * u_stride];
            block_hx5[lane] = hx[ev[lane * N_SHAPE + 5] * h_stride];
            block_uy5[lane] = uy[ev[lane * N_SHAPE + 5] * u_stride];
            block_hy5[lane] = hy[ev[lane * N_SHAPE + 5] * h_stride];
            block_outx0[lane] = scalar_t(0);
            block_outy0[lane] = scalar_t(0);
            block_outx1[lane] = scalar_t(0);
            block_outy1[lane] = scalar_t(0);
            block_outx2[lane] = scalar_t(0);
            block_outy2[lane] = scalar_t(0);
            block_outx3[lane] = scalar_t(0);
            block_outy3[lane] = scalar_t(0);
            block_outx4[lane] = scalar_t(0);
            block_outy4[lane] = scalar_t(0);
            block_outx5[lane] = scalar_t(0);
            block_outy5[lane] = scalar_t(0);
        }

        const scalar_t *const block_u_streams[N_SHAPE * 2] = {block_ux0, block_uy0, block_ux1, block_uy1, block_ux2, block_uy2, block_ux3, block_uy3, block_ux4, block_uy4, block_ux5, block_uy5};
        const scalar_t *const block_h_streams[N_SHAPE * 2] = {block_hx0, block_hy0, block_hx1, block_hy1, block_hx2, block_hy2, block_hx3, block_hy3, block_hx4, block_hy4, block_hx5, block_hy5};
        scalar_t *const block_out_streams[N_SHAPE * 2] = {block_outx0, block_outy0, block_outx1, block_outy1, block_outx2, block_outy2, block_outx3, block_outy3, block_outx4, block_outy4, block_outx5, block_outy5};

        const scalar_t *const block_coordinate_streams[N_SHAPE * 2] = {block_x0, block_y0, block_x1, block_y1, block_x2, block_y2, block_x3, block_y3, block_x4, block_y4, block_x5, block_y5};

        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t J00 = scalar_t(0);
                scalar_t J01 = scalar_t(0);
                scalar_t J10 = scalar_t(0);
                scalar_t J11 = scalar_t(0);
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                    J00 += block_coordinate_streams[shape * 2 + 0][lane] * g0;
                    J01 += block_coordinate_streams[shape * 2 + 0][lane] * g1;
                    J10 += block_coordinate_streams[shape * 2 + 1][lane] * g0;
                    J11 += block_coordinate_streams[shape * 2 + 1][lane] * g1;
                }
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = J11;
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = -J01;
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = -J10;
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = J00;
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * J11 - J01 * J10;
            }
        }

        generated_neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, grad_ref_x, grad_ref_y, q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 0] * out_stride] += block_outx0[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 0] * out_stride] += block_outy0[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 1] * out_stride] += block_outx1[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 1] * out_stride] += block_outy1[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 2] * out_stride] += block_outx2[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 2] * out_stride] += block_outy2[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 3] * out_stride] += block_outx3[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 3] * out_stride] += block_outy3[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 4] * out_stride] += block_outx4[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 4] * out_stride] += block_outy4[lane];

#pragma omp atomic update
            outx[ev[lane * N_SHAPE + 5] * out_stride] += block_outx5[lane];

#pragma omp atomic update
            outy[ev[lane * N_SHAPE + 5] * out_stride] += block_outy5[lane];

        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

