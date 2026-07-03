#include <type_traits>
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"
#include "../laplace_d2_tensor_product_local.hpp"

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
        SFEM_INLINE const scalar_t *affine_geometry_stream(const int,
                                                           const jacobian_t *const SFEM_RESTRICT source,
                                                           scalar_t *const                       SFEM_RESTRICT,
                                                           std::true_type) {
            return source;
        }

        template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
        SFEM_INLINE const scalar_t *affine_geometry_stream(const int                             nelems,
                                                           const jacobian_t *const SFEM_RESTRICT source,
                                                           scalar_t *const SFEM_RESTRICT         converted,
                                                           std::false_type) {
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                converted[lane] = scalar_t(source[lane]);
            }
            return converted;
        }

    }  // namespace codegen
}  // namespace sfem

namespace sfem {
    namespace codegen {

        template <typename scalar_t>
        struct laplace_quad4_affine_reference_data {
            static const scalar_t *shape_1d() {
                static const scalar_t data[4] = {scalar_t(0.78867513459481287),
                                                 scalar_t(0.21132486540518708),
                                                 scalar_t(0.21132486540518713),
                                                 scalar_t(0.78867513459481287)};
                return data;
            }
            static const scalar_t *grad_1d() {
                static const scalar_t data[4] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
                return data;
            }
            static const scalar_t *q_weight_1d() {
                static const scalar_t data[2] = {scalar_t(0.5), scalar_t(0.5)};
                return data;
            }
        };

        template <typename scalar_t>
        struct laplace_quad4_isoparametric_reference_data {
            static const scalar_t *shape_1d() {
                static const scalar_t data[4] = {scalar_t(0.78867513459481287),
                                                 scalar_t(0.21132486540518708),
                                                 scalar_t(0.21132486540518713),
                                                 scalar_t(0.78867513459481287)};
                return data;
            }
            static const scalar_t *grad_1d() {
                static const scalar_t data[4] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
                return data;
            }
            static const scalar_t *q_weight_1d() {
                static const scalar_t data[2] = {scalar_t(0.5), scalar_t(0.5)};
                return data;
            }
        };

    }  // namespace codegen
}  // namespace sfem

namespace sfem {
    namespace codegen {

        static const KernelDiagnostics laplace_quad4_residual_element_soa_diagnostics_data = {
                "laplace_quad4_residual_element_soa",
                "QUAD4",
                2,
                4,
                4,
                16,
                2,
                1,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                5,
                1,
                4,
                0,
                0,
                0,
                5,
                5,
                8,
                2,
                1,
                4,
                0,
                4,
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
                1.0};

    }  // namespace codegen
}  // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_quad4_residual_element_soa_arithmetic_intensity(const ptrdiff_t nelements,
                                                                          const size_t    scalar_bytes,
                                                                          const size_t    real_bytes,
                                                                          const size_t    accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
            nelements,
            scalar_bytes,
            real_bytes,
            accumulator_bytes);
}

extern "C" void laplace_quad4_residual_element_soa_print_rate(const double    elapsed,
                                                              const ptrdiff_t nelements,
                                                              const ptrdiff_t ndofs,
                                                              const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_residual_element_soa",
                                                &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(double),
                                                sizeof(double),
                                                sizeof(double));
}

extern "C" void laplace_quad4_residual_element_soa_float_print_rate(const double    elapsed,
                                                                    const ptrdiff_t nelements,
                                                                    const ptrdiff_t ndofs,
                                                                    const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_residual_element_soa_float",
                                                &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(float),
                                                sizeof(float),
                                                sizeof(float));
}

extern "C" void laplace_quad4_residual_affine_mesh_soa_print_rate(const double    elapsed,
                                                                  const ptrdiff_t nelements,
                                                                  const ptrdiff_t ndofs,
                                                                  const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh("laplace_quad4_residual_affine_mesh_soa",
                                                            &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
                                                            elapsed,
                                                            nelements,
                                                            ndofs,
                                                            repeat,
                                                            sizeof(double),
                                                            sizeof(double),
                                                            sizeof(double));
}

extern "C" void laplace_quad4_residual_affine_mesh_soa_float_print_rate(const double    elapsed,
                                                                        const ptrdiff_t nelements,
                                                                        const ptrdiff_t ndofs,
                                                                        const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh("laplace_quad4_residual_affine_mesh_soa_float",
                                                            &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
                                                            elapsed,
                                                            nelements,
                                                            ndofs,
                                                            repeat,
                                                            sizeof(float),
                                                            sizeof(float),
                                                            sizeof(float));
}

extern "C" void laplace_quad4_residual_isoparametric_mesh_soa_print_rate(const double    elapsed,
                                                                         const ptrdiff_t nelements,
                                                                         const ptrdiff_t ndofs,
                                                                         const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_quad4_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(double),
            sizeof(double),
            sizeof(double));
}

extern "C" void laplace_quad4_residual_isoparametric_mesh_soa_float_print_rate(const double    elapsed,
                                                                               const ptrdiff_t nelements,
                                                                               const ptrdiff_t ndofs,
                                                                               const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_quad4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_quad4_residual_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(float),
            sizeof(float),
            sizeof(float));
}

namespace sfem {
    namespace codegen {

        static const KernelDiagnostics laplace_quad4_jacobian_u_u_diagnostics_data = {"laplace_quad4_jacobian_u_u",
                                                                                      "QUAD4",
                                                                                      2,
                                                                                      4,
                                                                                      4,
                                                                                      16,
                                                                                      2,
                                                                                      1,
                                                                                      3,
                                                                                      0,
                                                                                      0,
                                                                                      0,
                                                                                      0,
                                                                                      0,
                                                                                      0,
                                                                                      5,
                                                                                      1,
                                                                                      4,
                                                                                      0,
                                                                                      0,
                                                                                      0,
                                                                                      5,
                                                                                      5,
                                                                                      8,
                                                                                      2,
                                                                                      1,
                                                                                      0,
                                                                                      4,
                                                                                      4,
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
                                                                                      1.0};

    }  // namespace codegen
}  // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_quad4_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_quad4_jacobian_u_u_arithmetic_intensity(const ptrdiff_t nelements,
                                                                  const size_t    scalar_bytes,
                                                                  const size_t    real_bytes,
                                                                  const size_t    accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_quad4_jacobian_u_u_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_quad4_jacobian_u_u_print_rate(const double    elapsed,
                                                      const ptrdiff_t nelements,
                                                      const ptrdiff_t ndofs,
                                                      const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_jacobian_u_u",
                                                &sfem::codegen::laplace_quad4_jacobian_u_u_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(double),
                                                sizeof(double),
                                                sizeof(double));
}

extern "C" void laplace_quad4_jacobian_u_u_float_print_rate(const double    elapsed,
                                                            const ptrdiff_t nelements,
                                                            const ptrdiff_t ndofs,
                                                            const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_jacobian_u_u_float",
                                                &sfem::codegen::laplace_quad4_jacobian_u_u_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(float),
                                                sizeof(float),
                                                sizeof(float));
}

namespace sfem {
    namespace codegen {

        static const KernelDiagnostics laplace_quad4_jacobian_action_element_soa_diagnostics_data = {
                "laplace_quad4_jacobian_action_element_soa",
                "QUAD4",
                2,
                4,
                4,
                16,
                2,
                1,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                5,
                1,
                4,
                0,
                0,
                0,
                5,
                5,
                8,
                2,
                1,
                0,
                4,
                4,
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
                1.0};

    }  // namespace codegen
}  // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_quad4_jacobian_action_element_soa_arithmetic_intensity(const ptrdiff_t nelements,
                                                                                 const size_t    scalar_bytes,
                                                                                 const size_t    real_bytes,
                                                                                 const size_t    accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
            nelements,
            scalar_bytes,
            real_bytes,
            accumulator_bytes);
}

extern "C" void laplace_quad4_jacobian_action_element_soa_print_rate(const double    elapsed,
                                                                     const ptrdiff_t nelements,
                                                                     const ptrdiff_t ndofs,
                                                                     const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_jacobian_action_element_soa",
                                                &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(double),
                                                sizeof(double),
                                                sizeof(double));
}

extern "C" void laplace_quad4_jacobian_action_element_soa_float_print_rate(const double    elapsed,
                                                                           const ptrdiff_t nelements,
                                                                           const ptrdiff_t ndofs,
                                                                           const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate("laplace_quad4_jacobian_action_element_soa_float",
                                                &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
                                                elapsed,
                                                nelements,
                                                ndofs,
                                                repeat,
                                                sizeof(float),
                                                sizeof(float),
                                                sizeof(float));
}

extern "C" void laplace_quad4_jacobian_action_affine_mesh_soa_print_rate(const double    elapsed,
                                                                         const ptrdiff_t nelements,
                                                                         const ptrdiff_t ndofs,
                                                                         const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_quad4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(double),
            sizeof(double),
            sizeof(double));
}

extern "C" void laplace_quad4_jacobian_action_affine_mesh_soa_float_print_rate(const double    elapsed,
                                                                               const ptrdiff_t nelements,
                                                                               const ptrdiff_t ndofs,
                                                                               const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_quad4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(float),
            sizeof(float),
            sizeof(float));
}

extern "C" void laplace_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(const double    elapsed,
                                                                                const ptrdiff_t nelements,
                                                                                const ptrdiff_t ndofs,
                                                                                const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_quad4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(double),
            sizeof(double),
            sizeof(double));
}

extern "C" void laplace_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(const double    elapsed,
                                                                                      const ptrdiff_t nelements,
                                                                                      const ptrdiff_t ndofs,
                                                                                      const int       repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_quad4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_quad4_jacobian_action_element_soa_diagnostics_data,
            elapsed,
            nelements,
            ndofs,
            repeat,
            sizeof(float),
            sizeof(float),
            sizeof(float));
}

extern "C" int laplace_quad4_residual_element_soa(const int                         nelems,
                                                  const ptrdiff_t                   geometry_stride,
                                                  const double *const SFEM_RESTRICT determinant,
                                                  const double *const SFEM_RESTRICT adjugate[4],
                                                  const double *const SFEM_RESTRICT current[4],
                                                  const double                      kappa,
                                                  double *const SFEM_RESTRICT       output[4]) {
    sfem::codegen::laplace_d2_tensor_product_residual_block<double, 4, 4, 16>(
            nelems,
            geometry_stride,
            determinant,
            adjugate,
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::shape_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::grad_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::q_weight_1d(),
            current,
            kappa,
            output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_quad4_residual_element_soa_float(const int                        nelems,
                                                        const ptrdiff_t                  geometry_stride,
                                                        const float *const SFEM_RESTRICT determinant,
                                                        const float *const SFEM_RESTRICT adjugate[4],
                                                        const float *const SFEM_RESTRICT current[4],
                                                        const float                      kappa,
                                                        float *const SFEM_RESTRICT       output[4]) {
    sfem::codegen::laplace_d2_tensor_product_residual_block<float, 4, 4, 16>(
            nelems,
            geometry_stride,
            determinant,
            adjugate,
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::shape_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::grad_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::q_weight_1d(),
            current,
            kappa,
            output);
    return SFEM_SUCCESS;
}

namespace sfem {
    namespace codegen {

        template <typename scalar_t, typename jacobian_t>
        static SFEM_INLINE int laplace_quad4_residual_affine_mesh_soa_impl(
                const ptrdiff_t                       nelements,
                const ptrdiff_t                       nnodes,
                idx_t **const SFEM_RESTRICT           elements,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
                const scalar_t                        kappa,
                const ptrdiff_t                       current_stride,
                const scalar_t *const SFEM_RESTRICT   u,
                const ptrdiff_t                       out_stride,
                scalar_t *const SFEM_RESTRICT         u_out) {
            static constexpr int DIM         = 2;
            static constexpr int N_QP        = 4;
            static constexpr int N_SHAPE     = 4;
            static constexpr int N_FIELDS    = 1;
            static constexpr int VECTOR_SIZE = 16;
            (void)nnodes;
            const scalar_t *const affine_shape_1d = sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::shape_1d();
            const scalar_t *const affine_grad_1d  = sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::grad_1d();
            const scalar_t *const affine_q_weight_1d =
                    sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
            for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
                idx_t     ev[VECTOR_SIZE * N_SHAPE];
                scalar_t  block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
                    ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
                    ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
                    ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_current[0][lane] = u[ev[0 * VECTOR_SIZE + lane] * current_stride];
                    block_current[1][lane] = u[ev[1 * VECTOR_SIZE + lane] * current_stride];
                    block_current[2][lane] = u[ev[2 * VECTOR_SIZE + lane] * current_stride];
                    block_current[3][lane] = u[ev[3 * VECTOR_SIZE + lane] * current_stride];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_output[0][lane] = scalar_t(0);
                    block_output[1][lane] = scalar_t(0);
                    block_output[2][lane] = scalar_t(0);
                    block_output[3][lane] = scalar_t(0);
                }

                const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {
                        block_current[0], block_current[1], block_current[3], block_current[2]};
                scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {
                        block_output[0], block_output[1], block_output[3], block_output[2]};
                scalar_t              block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate0 + evbegin,
                                                                                  block_jacobian_adjugate0_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate1 + evbegin,
                                                                                  block_jacobian_adjugate1_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate2 + evbegin,
                                                                                  block_jacobian_adjugate2_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate3 + evbegin,
                                                                                  block_jacobian_adjugate3_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_determinant0 + evbegin,
                                                                                  block_jacobian_determinant0_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                const scalar_t *const block_adjugate[4] = {
                        block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};

                laplace_d2_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems,
                                                                                               0,
                                                                                               block_jacobian_determinant0,
                                                                                               block_adjugate,
                                                                                               affine_shape_1d,
                                                                                               affine_grad_1d,
                                                                                               affine_q_weight_1d,
                                                                                               block_current_streams,
                                                                                               kappa,
                                                                                               block_output_streams);

                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

    }  // namespace codegen
}  // namespace sfem

extern "C" int laplace_quad4_residual_affine_mesh_soa(const ptrdiff_t                   nelements,
                                                      const ptrdiff_t                   nnodes,
                                                      idx_t **const SFEM_RESTRICT       elements,
                                                      const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                                                      const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                                                      const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                                                      const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                                                      const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
                                                      const double                      kappa,
                                                      const ptrdiff_t                   current_stride,
                                                      const double *const SFEM_RESTRICT u,
                                                      const ptrdiff_t                   out_stride,
                                                      double *const SFEM_RESTRICT       u_out) {
    return sfem::codegen::laplace_quad4_residual_affine_mesh_soa_impl<double, geom_t>(nelements,
                                                                                      nnodes,
                                                                                      elements,
                                                                                      g_jacobian_adjugate0,
                                                                                      g_jacobian_adjugate1,
                                                                                      g_jacobian_adjugate2,
                                                                                      g_jacobian_adjugate3,
                                                                                      g_jacobian_determinant0,
                                                                                      kappa,
                                                                                      current_stride,
                                                                                      u,
                                                                                      out_stride,
                                                                                      u_out);
}

extern "C" int laplace_quad4_residual_affine_mesh_soa_float(const ptrdiff_t                   nelements,
                                                            const ptrdiff_t                   nnodes,
                                                            idx_t **const SFEM_RESTRICT       elements,
                                                            const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                                                            const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                                                            const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                                                            const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                                                            const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
                                                            const float                       kappa,
                                                            const ptrdiff_t                   current_stride,
                                                            const float *const SFEM_RESTRICT  u,
                                                            const ptrdiff_t                   out_stride,
                                                            float *const SFEM_RESTRICT        u_out) {
    return sfem::codegen::laplace_quad4_residual_affine_mesh_soa_impl<float, geom_t>(nelements,
                                                                                     nnodes,
                                                                                     elements,
                                                                                     g_jacobian_adjugate0,
                                                                                     g_jacobian_adjugate1,
                                                                                     g_jacobian_adjugate2,
                                                                                     g_jacobian_adjugate3,
                                                                                     g_jacobian_determinant0,
                                                                                     kappa,
                                                                                     current_stride,
                                                                                     u,
                                                                                     out_stride,
                                                                                     u_out);
}

namespace sfem {
    namespace codegen {

        template <typename scalar_t>
        static SFEM_INLINE int laplace_quad4_residual_isoparametric_mesh_soa_impl(const ptrdiff_t             nelements,
                                                                                  const ptrdiff_t             nnodes,
                                                                                  idx_t **const SFEM_RESTRICT elements,
                                                                                  const geom_t *const *const SFEM_RESTRICT points,
                                                                                  const scalar_t                           kappa,
                                                                                  const ptrdiff_t current_stride,
                                                                                  const scalar_t *const SFEM_RESTRICT u,
                                                                                  const ptrdiff_t                     out_stride,
                                                                                  scalar_t *const SFEM_RESTRICT       u_out) {
            static constexpr int DIM         = 2;
            static constexpr int N_QP        = 4;
            static constexpr int N_SHAPE     = 4;
            static constexpr int N_FIELDS    = 1;
            static constexpr int VECTOR_SIZE = 16;
            (void)nnodes;
            const scalar_t *const isoparametric_shape_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::shape_1d();
            const scalar_t *const isoparametric_grad_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::grad_1d();
            const scalar_t *const isoparametric_q_weight_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
            for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
                idx_t     ev[VECTOR_SIZE * N_SHAPE];
                scalar_t  block_coordinates[2 * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_adjugate_data[4][N_QP * VECTOR_SIZE];
                scalar_t  block_determinant[N_QP * VECTOR_SIZE];
                scalar_t  block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
                    ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
                    ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
                    ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinates[0][lane] = points[0][ev[0 * VECTOR_SIZE + lane]];
                    block_coordinates[1][lane] = points[1][ev[0 * VECTOR_SIZE + lane]];
                    block_current[0][lane]     = u[ev[0 * VECTOR_SIZE + lane] * current_stride];
                    block_coordinates[2][lane] = points[0][ev[1 * VECTOR_SIZE + lane]];
                    block_coordinates[3][lane] = points[1][ev[1 * VECTOR_SIZE + lane]];
                    block_current[1][lane]     = u[ev[1 * VECTOR_SIZE + lane] * current_stride];
                    block_coordinates[4][lane] = points[0][ev[2 * VECTOR_SIZE + lane]];
                    block_coordinates[5][lane] = points[1][ev[2 * VECTOR_SIZE + lane]];
                    block_current[2][lane]     = u[ev[2 * VECTOR_SIZE + lane] * current_stride];
                    block_coordinates[6][lane] = points[0][ev[3 * VECTOR_SIZE + lane]];
                    block_coordinates[7][lane] = points[1][ev[3 * VECTOR_SIZE + lane]];
                    block_current[3][lane]     = u[ev[3 * VECTOR_SIZE + lane] * current_stride];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_output[0][lane] = scalar_t(0);
                    block_output[1][lane] = scalar_t(0);
                    block_output[2][lane] = scalar_t(0);
                    block_output[3][lane] = scalar_t(0);
                }

                const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0],
                                                                                 block_coordinates[1],
                                                                                 block_coordinates[2],
                                                                                 block_coordinates[3],
                                                                                 block_coordinates[6],
                                                                                 block_coordinates[7],
                                                                                 block_coordinates[4],
                                                                                 block_coordinates[5]};
                scalar_t              coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
                scalar_t              coordinate_value[DIM * N_QP * VECTOR_SIZE];
                tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(nelems,
                                                                                isoparametric_shape_1d,
                                                                                isoparametric_grad_1d,
                                                                                block_coordinate_streams,
                                                                                coordinate_value,
                                                                                coordinate_grad_ref);

                scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {
                        block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
                geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                        nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

                const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {
                        block_current[0], block_current[1], block_current[3], block_current[2]};
                scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {
                        block_output[0], block_output[1], block_output[3], block_output[2]};
                const scalar_t *const block_adjugate[4] = {
                        block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

                laplace_d2_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems,
                                                                                               VECTOR_SIZE,
                                                                                               block_determinant,
                                                                                               block_adjugate,
                                                                                               isoparametric_shape_1d,
                                                                                               isoparametric_grad_1d,
                                                                                               isoparametric_q_weight_1d,
                                                                                               block_current_streams,
                                                                                               kappa,
                                                                                               block_output_streams);

                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

    }  // namespace codegen
}  // namespace sfem

extern "C" int laplace_quad4_residual_isoparametric_mesh_soa(const ptrdiff_t                          nelements,
                                                             const ptrdiff_t                          nnodes,
                                                             idx_t **const SFEM_RESTRICT              elements,
                                                             const geom_t *const *const SFEM_RESTRICT points,
                                                             const double                             kappa,
                                                             const ptrdiff_t                          current_stride,
                                                             const double *const SFEM_RESTRICT        u,
                                                             const ptrdiff_t                          out_stride,
                                                             double *const SFEM_RESTRICT              u_out) {
    return sfem::codegen::laplace_quad4_residual_isoparametric_mesh_soa_impl<double>(
            nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_quad4_residual_isoparametric_mesh_soa_float(const ptrdiff_t                          nelements,
                                                                   const ptrdiff_t                          nnodes,
                                                                   idx_t **const SFEM_RESTRICT              elements,
                                                                   const geom_t *const *const SFEM_RESTRICT points,
                                                                   const float                              kappa,
                                                                   const ptrdiff_t                          current_stride,
                                                                   const float *const SFEM_RESTRICT         u,
                                                                   const ptrdiff_t                          out_stride,
                                                                   float *const SFEM_RESTRICT               u_out) {
    return sfem::codegen::laplace_quad4_residual_isoparametric_mesh_soa_impl<float>(
            nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_quad4_residual_isoparametric_mesh_aos(const ptrdiff_t                          nelements,
                                                             const ptrdiff_t                          nnodes,
                                                             idx_t **const SFEM_RESTRICT              elements,
                                                             const geom_t *const *const SFEM_RESTRICT points,
                                                             const double *const SFEM_RESTRICT        parameters,
                                                             const double *const SFEM_RESTRICT        current,
                                                             double *const SFEM_RESTRICT              output) {
    return laplace_quad4_residual_isoparametric_mesh_soa(
            nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_quad4_residual_isoparametric_mesh_aos_float(const ptrdiff_t                          nelements,
                                                                   const ptrdiff_t                          nnodes,
                                                                   idx_t **const SFEM_RESTRICT              elements,
                                                                   const geom_t *const *const SFEM_RESTRICT points,
                                                                   const float *const SFEM_RESTRICT         parameters,
                                                                   const float *const SFEM_RESTRICT         current,
                                                                   float *const SFEM_RESTRICT               output) {
    return laplace_quad4_residual_isoparametric_mesh_soa_float(
            nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_quad4_jacobian_action_element_soa(const int                         nelems,
                                                         const ptrdiff_t                   geometry_stride,
                                                         const double *const SFEM_RESTRICT determinant,
                                                         const double *const SFEM_RESTRICT adjugate[4],
                                                         const double *const SFEM_RESTRICT direction[4],
                                                         const double                      kappa,
                                                         double *const SFEM_RESTRICT       output[4]) {
    sfem::codegen::laplace_d2_tensor_product_jacobian_action_block<double, 4, 4, 16>(
            nelems,
            geometry_stride,
            determinant,
            adjugate,
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::shape_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::grad_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<double>::q_weight_1d(),
            direction,
            kappa,
            output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_quad4_jacobian_action_element_soa_float(const int                        nelems,
                                                               const ptrdiff_t                  geometry_stride,
                                                               const float *const SFEM_RESTRICT determinant,
                                                               const float *const SFEM_RESTRICT adjugate[4],
                                                               const float *const SFEM_RESTRICT direction[4],
                                                               const float                      kappa,
                                                               float *const SFEM_RESTRICT       output[4]) {
    sfem::codegen::laplace_d2_tensor_product_jacobian_action_block<float, 4, 4, 16>(
            nelems,
            geometry_stride,
            determinant,
            adjugate,
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::shape_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::grad_1d(),
            sfem::codegen::laplace_quad4_isoparametric_reference_data<float>::q_weight_1d(),
            direction,
            kappa,
            output);
    return SFEM_SUCCESS;
}

namespace sfem {
    namespace codegen {

        template <typename scalar_t, typename jacobian_t>
        static SFEM_INLINE int laplace_quad4_jacobian_action_affine_mesh_soa_impl(
                const ptrdiff_t                       nelements,
                const ptrdiff_t                       nnodes,
                idx_t **const SFEM_RESTRICT           elements,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
                const scalar_t                        kappa,
                const ptrdiff_t                       direction_stride,
                const scalar_t *const SFEM_RESTRICT   u_direction,
                const ptrdiff_t                       out_stride,
                scalar_t *const SFEM_RESTRICT         u_out) {
            static constexpr int DIM         = 2;
            static constexpr int N_QP        = 4;
            static constexpr int N_SHAPE     = 4;
            static constexpr int N_FIELDS    = 1;
            static constexpr int VECTOR_SIZE = 16;
            (void)nnodes;
            const scalar_t *const affine_shape_1d = sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::shape_1d();
            const scalar_t *const affine_grad_1d  = sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::grad_1d();
            const scalar_t *const affine_q_weight_1d =
                    sfem::codegen::laplace_quad4_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
            for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
                idx_t     ev[VECTOR_SIZE * N_SHAPE];
                scalar_t  block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
                    ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
                    ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
                    ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_direction[0][lane] = u_direction[ev[0 * VECTOR_SIZE + lane] * direction_stride];
                    block_direction[1][lane] = u_direction[ev[1 * VECTOR_SIZE + lane] * direction_stride];
                    block_direction[2][lane] = u_direction[ev[2 * VECTOR_SIZE + lane] * direction_stride];
                    block_direction[3][lane] = u_direction[ev[3 * VECTOR_SIZE + lane] * direction_stride];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_output[0][lane] = scalar_t(0);
                    block_output[1][lane] = scalar_t(0);
                    block_output[2][lane] = scalar_t(0);
                    block_output[3][lane] = scalar_t(0);
                }

                const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {
                        block_direction[0], block_direction[1], block_direction[3], block_direction[2]};
                scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {
                        block_output[0], block_output[1], block_output[3], block_output[2]};
                scalar_t              block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate0 + evbegin,
                                                                                  block_jacobian_adjugate0_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate1 + evbegin,
                                                                                  block_jacobian_adjugate1_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate2 + evbegin,
                                                                                  block_jacobian_adjugate2_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_adjugate3 + evbegin,
                                                                                  block_jacobian_adjugate3_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                scalar_t              block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 =
                        affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(nelems,
                                                                                  g_jacobian_determinant0 + evbegin,
                                                                                  block_jacobian_determinant0_data,
                                                                                  std::is_same<jacobian_t, scalar_t>());
                const scalar_t *const block_adjugate[4] = {
                        block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};

                laplace_d2_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems,
                                                                                                      0,
                                                                                                      block_jacobian_determinant0,
                                                                                                      block_adjugate,
                                                                                                      affine_shape_1d,
                                                                                                      affine_grad_1d,
                                                                                                      affine_q_weight_1d,
                                                                                                      block_direction_streams,
                                                                                                      kappa,
                                                                                                      block_output_streams);

                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

    }  // namespace codegen
}  // namespace sfem

extern "C" int laplace_quad4_jacobian_action_affine_mesh_soa(const ptrdiff_t                   nelements,
                                                             const ptrdiff_t                   nnodes,
                                                             idx_t **const SFEM_RESTRICT       elements,
                                                             const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                                                             const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                                                             const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                                                             const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                                                             const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
                                                             const double                      kappa,
                                                             const ptrdiff_t                   direction_stride,
                                                             const double *const SFEM_RESTRICT u_direction,
                                                             const ptrdiff_t                   out_stride,
                                                             double *const SFEM_RESTRICT       u_out) {
    return sfem::codegen::laplace_quad4_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements,
                                                                                             nnodes,
                                                                                             elements,
                                                                                             g_jacobian_adjugate0,
                                                                                             g_jacobian_adjugate1,
                                                                                             g_jacobian_adjugate2,
                                                                                             g_jacobian_adjugate3,
                                                                                             g_jacobian_determinant0,
                                                                                             kappa,
                                                                                             direction_stride,
                                                                                             u_direction,
                                                                                             out_stride,
                                                                                             u_out);
}

extern "C" int laplace_quad4_jacobian_action_affine_mesh_soa_float(const ptrdiff_t                   nelements,
                                                                   const ptrdiff_t                   nnodes,
                                                                   idx_t **const SFEM_RESTRICT       elements,
                                                                   const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
                                                                   const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
                                                                   const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
                                                                   const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
                                                                   const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
                                                                   const float                       kappa,
                                                                   const ptrdiff_t                   direction_stride,
                                                                   const float *const SFEM_RESTRICT  u_direction,
                                                                   const ptrdiff_t                   out_stride,
                                                                   float *const SFEM_RESTRICT        u_out) {
    return sfem::codegen::laplace_quad4_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements,
                                                                                            nnodes,
                                                                                            elements,
                                                                                            g_jacobian_adjugate0,
                                                                                            g_jacobian_adjugate1,
                                                                                            g_jacobian_adjugate2,
                                                                                            g_jacobian_adjugate3,
                                                                                            g_jacobian_determinant0,
                                                                                            kappa,
                                                                                            direction_stride,
                                                                                            u_direction,
                                                                                            out_stride,
                                                                                            u_out);
}

namespace sfem {
    namespace codegen {

        template <typename scalar_t>
        static SFEM_INLINE int laplace_quad4_jacobian_action_isoparametric_mesh_soa_impl(
                const ptrdiff_t                          nelements,
                const ptrdiff_t                          nnodes,
                idx_t **const SFEM_RESTRICT              elements,
                const geom_t *const *const SFEM_RESTRICT points,
                const scalar_t                           kappa,
                const ptrdiff_t                          direction_stride,
                const scalar_t *const SFEM_RESTRICT      u_direction,
                const ptrdiff_t                          out_stride,
                scalar_t *const SFEM_RESTRICT            u_out) {
            static constexpr int DIM         = 2;
            static constexpr int N_QP        = 4;
            static constexpr int N_SHAPE     = 4;
            static constexpr int N_FIELDS    = 1;
            static constexpr int VECTOR_SIZE = 16;
            (void)nnodes;
            const scalar_t *const isoparametric_shape_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::shape_1d();
            const scalar_t *const isoparametric_grad_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::grad_1d();
            const scalar_t *const isoparametric_q_weight_1d =
                    sfem::codegen::laplace_quad4_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
            for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
                idx_t     ev[VECTOR_SIZE * N_SHAPE];
                scalar_t  block_coordinates[2 * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_adjugate_data[4][N_QP * VECTOR_SIZE];
                scalar_t  block_determinant[N_QP * VECTOR_SIZE];
                scalar_t  block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
                scalar_t  block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    ev[0 * VECTOR_SIZE + lane] = elements[0][evbegin + lane];
                    ev[1 * VECTOR_SIZE + lane] = elements[1][evbegin + lane];
                    ev[2 * VECTOR_SIZE + lane] = elements[2][evbegin + lane];
                    ev[3 * VECTOR_SIZE + lane] = elements[3][evbegin + lane];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinates[0][lane] = points[0][ev[0 * VECTOR_SIZE + lane]];
                    block_coordinates[1][lane] = points[1][ev[0 * VECTOR_SIZE + lane]];
                    block_direction[0][lane]   = u_direction[ev[0 * VECTOR_SIZE + lane] * direction_stride];
                    block_coordinates[2][lane] = points[0][ev[1 * VECTOR_SIZE + lane]];
                    block_coordinates[3][lane] = points[1][ev[1 * VECTOR_SIZE + lane]];
                    block_direction[1][lane]   = u_direction[ev[1 * VECTOR_SIZE + lane] * direction_stride];
                    block_coordinates[4][lane] = points[0][ev[2 * VECTOR_SIZE + lane]];
                    block_coordinates[5][lane] = points[1][ev[2 * VECTOR_SIZE + lane]];
                    block_direction[2][lane]   = u_direction[ev[2 * VECTOR_SIZE + lane] * direction_stride];
                    block_coordinates[6][lane] = points[0][ev[3 * VECTOR_SIZE + lane]];
                    block_coordinates[7][lane] = points[1][ev[3 * VECTOR_SIZE + lane]];
                    block_direction[3][lane]   = u_direction[ev[3 * VECTOR_SIZE + lane] * direction_stride];
                }

#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_output[0][lane] = scalar_t(0);
                    block_output[1][lane] = scalar_t(0);
                    block_output[2][lane] = scalar_t(0);
                    block_output[3][lane] = scalar_t(0);
                }

                const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0],
                                                                                 block_coordinates[1],
                                                                                 block_coordinates[2],
                                                                                 block_coordinates[3],
                                                                                 block_coordinates[6],
                                                                                 block_coordinates[7],
                                                                                 block_coordinates[4],
                                                                                 block_coordinates[5]};
                scalar_t              coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
                scalar_t              coordinate_value[DIM * N_QP * VECTOR_SIZE];
                tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(nelems,
                                                                                isoparametric_shape_1d,
                                                                                isoparametric_grad_1d,
                                                                                block_coordinate_streams,
                                                                                coordinate_value,
                                                                                coordinate_grad_ref);

                scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {
                        block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
                geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                        nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

                const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {
                        block_direction[0], block_direction[1], block_direction[3], block_direction[2]};
                scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {
                        block_output[0], block_output[1], block_output[3], block_output[2]};
                const scalar_t *const block_adjugate[4] = {
                        block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

                laplace_d2_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems,
                                                                                                      VECTOR_SIZE,
                                                                                                      block_determinant,
                                                                                                      block_adjugate,
                                                                                                      isoparametric_shape_1d,
                                                                                                      isoparametric_grad_1d,
                                                                                                      isoparametric_q_weight_1d,
                                                                                                      block_direction_streams,
                                                                                                      kappa,
                                                                                                      block_output_streams);

                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[0 * VECTOR_SIZE + scatter] * out_stride] += block_output[0][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[1 * VECTOR_SIZE + scatter] * out_stride] += block_output[1][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[2 * VECTOR_SIZE + scatter] * out_stride] += block_output[2][scatter];
                    }
                }
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                        u_out[ev[3 * VECTOR_SIZE + scatter] * out_stride] += block_output[3][scatter];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

    }  // namespace codegen
}  // namespace sfem

extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_soa(const ptrdiff_t                          nelements,
                                                                    const ptrdiff_t                          nnodes,
                                                                    idx_t **const SFEM_RESTRICT              elements,
                                                                    const geom_t *const *const SFEM_RESTRICT points,
                                                                    const double                             kappa,
                                                                    const ptrdiff_t                          direction_stride,
                                                                    const double *const SFEM_RESTRICT        u_direction,
                                                                    const ptrdiff_t                          out_stride,
                                                                    double *const SFEM_RESTRICT              u_out) {
    return sfem::codegen::laplace_quad4_jacobian_action_isoparametric_mesh_soa_impl<double>(
            nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_soa_float(const ptrdiff_t                          nelements,
                                                                          const ptrdiff_t                          nnodes,
                                                                          idx_t **const SFEM_RESTRICT              elements,
                                                                          const geom_t *const *const SFEM_RESTRICT points,
                                                                          const float                              kappa,
                                                                          const ptrdiff_t                  direction_stride,
                                                                          const float *const SFEM_RESTRICT u_direction,
                                                                          const ptrdiff_t                  out_stride,
                                                                          float *const SFEM_RESTRICT       u_out) {
    return sfem::codegen::laplace_quad4_jacobian_action_isoparametric_mesh_soa_impl<float>(
            nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_aos(const ptrdiff_t                          nelements,
                                                                    const ptrdiff_t                          nnodes,
                                                                    idx_t **const SFEM_RESTRICT              elements,
                                                                    const geom_t *const *const SFEM_RESTRICT points,
                                                                    const double *const SFEM_RESTRICT        parameters,
                                                                    const double *const SFEM_RESTRICT        direction,
                                                                    double *const SFEM_RESTRICT              output) {
    return laplace_quad4_jacobian_action_isoparametric_mesh_soa(
            nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_aos_float(const ptrdiff_t                          nelements,
                                                                          const ptrdiff_t                          nnodes,
                                                                          idx_t **const SFEM_RESTRICT              elements,
                                                                          const geom_t *const *const SFEM_RESTRICT points,
                                                                          const float *const SFEM_RESTRICT         parameters,
                                                                          const float *const SFEM_RESTRICT         direction,
                                                                          float *const SFEM_RESTRICT               output) {
    return laplace_quad4_jacobian_action_isoparametric_mesh_soa_float(
            nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
