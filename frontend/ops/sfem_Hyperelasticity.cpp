#include "sfem_Hyperelasticity.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_Env.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include "generic_hyperelasticity.hpp"

// HEX8 dedicated micro-kernels and helpers
#include "hex8_inline_cpu.h"
#include "line_quadrature.h"
#include "hex8_partial_assembly_neohookean_inline.h"
#include "hex8_neohookean_ogden_local.h"
#include "neohookean_ogden.h"

#include <dlfcn.h>
#include <math.h>
#include <mpi.h>
#include <functional>
#include <vector>

namespace sfem {

    static constexpr int IKMN_SIZE = 45;

    struct HyperelasticityAssemblyData {
        SharedBuffer<metric_tensor_t> partial_assembly_buffer;
        SharedBuffer<scaling_t>       compression_scaling;
        SharedBuffer<compressed_t>    partial_assembly_compressed;
        SharedBuffer<idx_t *>         elements;
        ptrdiff_t                     elements_stride{1};

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};

        int compress_partial_assembly(const OpDomain &domain) {
            auto mesh = domain.block;

            if (use_compression) {
                if (!compression_scaling) {
                    compression_scaling         = sfem::create_host_buffer<scaling_t>(mesh->n_elements());
                    partial_assembly_compressed = sfem::create_host_buffer<compressed_t>(mesh->n_elements() * IKMN_SIZE);
                }

                auto      cs         = compression_scaling->data();
                auto      pa         = partial_assembly_buffer->data();
                auto      pac        = partial_assembly_compressed->data();
                ptrdiff_t n_elements = mesh->n_elements();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai = &pa[i * IKMN_SIZE];
                    cs[i]    = pai[0];
                    for (int v = 1; v < IKMN_SIZE; v++) {
                        cs[i] = MAX(cs[i], fabs(pai[v]));
                    }
                }

                real_t max_scaling = 0;

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    if (cs[i] > real_t(FP16_MAX)) {
                        max_scaling = MAX(max_scaling, cs[i]);
                        cs[i]       = real_t(cs[i] + 1e-8) / real_t(FP16_MAX);
                    } else {
                        cs[i] = 1;
                    }
                }

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai  = &pa[i * IKMN_SIZE];
                    auto paci = &pac[i * IKMN_SIZE];
                    for (int v = 0; v < IKMN_SIZE; v++) {
                        paci[v] = (compressed_t)(pai[v] / cs[i]);

                        assert(cs[i] > 0);
                        // assert(std::isfinite(paci[v]));
                    }
                }
            }

            return SFEM_SUCCESS;
        }
    };

    struct HyperelasticityKernels {
        virtual int hessian_aos(const ptrdiff_t,
                                const ptrdiff_t,
                                idx_t **const,
                                geom_t **const,
                                const real_t *const,
                                const count_t *const,
                                const idx_t *const,
                                real_t *const) = 0;

        virtual int hessian_diag_aos(const ptrdiff_t,
                                     const ptrdiff_t,
                                     idx_t **const,
                                     geom_t **const,
                                     const real_t *const,
                                     real_t *const) = 0;

        virtual int gradient_aos(const ptrdiff_t,
                                 const ptrdiff_t,
                                 idx_t **const,
                                 geom_t **const,
                                 const real_t *const,
                                 real_t *const) = 0;

        virtual int apply_aos(const ptrdiff_t,
                              const ptrdiff_t,
                              idx_t **const,
                              geom_t **const,
                              const real_t *const,
                              const real_t *const,
                              real_t *const) = 0;

        virtual int partial_assembly_apply(const ptrdiff_t,
                                           const ptrdiff_t,
                                           idx_t **const,
                                           const metric_tensor_t *const,
                                           const ptrdiff_t,
                                           const real_t *const,
                                           const real_t *const,
                                           const real_t *const,
                                           const ptrdiff_t,
                                           real_t *const,
                                           real_t *const,
                                           real_t *const) = 0;

        virtual int compressed_partial_assembly_apply(const ptrdiff_t,
                                                      const ptrdiff_t,
                                                      idx_t **const,
                                                      const compressed_t *const,
                                                      const scaling_t *const,
                                                      const ptrdiff_t,
                                                      const real_t *const,
                                                      const real_t *const,
                                                      const real_t *const,
                                                      const ptrdiff_t,
                                                      real_t *const,
                                                      real_t *const,
                                                      real_t *const) = 0;

        virtual int hessian_partial_assembly(const ptrdiff_t,
                                             const ptrdiff_t,
                                             idx_t **const,
                                             geom_t **const,
                                             const ptrdiff_t,
                                             const real_t *const,
                                             const real_t *const,
                                             const real_t *const,
                                             metric_tensor_t *const) = 0;

        // Objective and path-evaluated objective (HEX8 specialized forms)
        virtual int objective(const ptrdiff_t,
                              const ptrdiff_t,
                              const ptrdiff_t,
                              idx_t **const,
                              geom_t **const,
                              const ptrdiff_t,
                              const real_t *const,
                              const real_t *const,
                              const real_t *const,
                              const int,
                              real_t *const) = 0;

        virtual int objective_steps(const ptrdiff_t,
                                    const ptrdiff_t,
                                    const ptrdiff_t,
                                    idx_t **const,
                                    geom_t **const,
                                    const ptrdiff_t,
                                    const real_t *const,
                                    const real_t *const,
                                    const real_t *const,
                                    const ptrdiff_t,
                                    const real_t *const,
                                    const real_t *const,
                                    const real_t *const,
                                    const int,
                                    const real_t *const,
                                    real_t *const) = 0;
        ~HyperelasticityKernels()                  = default;
    };

    namespace hex8 {
        class NeoHookeanOgden final : public HyperelasticityKernels {
        public:
            // Implement using generic_hyperelasticity.hpp and HEX8 micro-kernels
            int hessian_aos(const ptrdiff_t        nelements,
                            const ptrdiff_t        nnodes,
                            idx_t **const          elements,
                            geom_t **const         points,
                            const real_t *const    u,
                            const count_t *const   rowptr,
                            const idx_t *const     colidx,
                            real_t *const          values) override {
                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));
                return neohookean_ogden_hessian_aos(
                        HEX8, nelements, nnodes, elements, points, mu, lambda, u, rowptr, colidx, values);
            }

            int hessian_diag_aos(const ptrdiff_t       nelements,
                                 const ptrdiff_t       nnodes,
                                 idx_t **const         elements,
                                 geom_t **const        points,
                                 const real_t *const   u,
                                 real_t *const         out) override {
                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                auto diag_micro = [mu, lambda](const scalar_t *lx,
                                               const scalar_t *ly,
                                               const scalar_t *lz,
                                               const scalar_t *ux,
                                               const scalar_t *uy,
                                               const scalar_t *uz,
                                               scalar_t *      ediag) {
                    static const int       n_qp = line_q2_n;
                    static const scalar_t *qx   = line_q2_x;
                    static const scalar_t *qw   = line_q2_w;
                    scalar_t               Jadj[9];
                    scalar_t               Jdet;
                    for (int kz = 0; kz < n_qp; ++kz) {
                        for (int ky = 0; ky < n_qp; ++ky) {
                            for (int kx = 0; kx < n_qp; ++kx) {
                                hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], Jadj, &Jdet);
                                scalar_t H_diagx[8] = {0};
                                scalar_t H_diagy[8] = {0};
                                scalar_t H_diagz[8] = {0};
                                hex8_neohookean_ogden_hessian_diag(Jadj,
                                                                   Jdet,
                                                                   qx[kx],
                                                                   qx[ky],
                                                                   qx[kz],
                                                                   qw[kx] * qw[ky] * qw[kz],
                                                                   mu,
                                                                   lambda,
                                                                   ux,
                                                                   uy,
                                                                   uz,
                                                                   H_diagx,
                                                                   H_diagy,
                                                                   H_diagz);
                                                                   
                                for (int a = 0; a < 8; ++a) {
                                    ediag[3 * a + 0] += H_diagx[a];
                                    ediag[3 * a + 1] += H_diagy[a];
                                    ediag[3 * a + 2] += H_diagz[a];
                                }
                            }
                        }
                    }
                };

                Hyperelasticity3 helper;
                return helper.hessian_diag<8>(diag_micro,
                                               nelements,
                                               /*stride*/ 1,
                                               nnodes,
                                               elements,
                                               points,
                                               /*u_stride*/ 3,
                                               &u[0],
                                               &u[1],
                                               &u[2],
                                               out);
            }

            int gradient_aos(const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             idx_t **const   elements,
                             geom_t **const  points,
                             const real_t *const u,
                             real_t *const       out) override {

                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                auto micro_kernel = [mu, lambda](const scalar_t *lx,
                                                 const scalar_t *ly,
                                                 const scalar_t *lz,
                                                 const scalar_t *ux,
                                                 const scalar_t *uy,
                                                 const scalar_t *uz,
                                                 scalar_t *      eoutx,
                                                 scalar_t *      eouty,
                                                 scalar_t *      eoutz) {
                    static const int       n_qp = line_q2_n;
                    static const scalar_t *qx   = line_q2_x;
                    static const scalar_t *qw   = line_q2_w;
                    scalar_t Jadj[9];
                    scalar_t Jdet;
                    for (int kz = 0; kz < n_qp; ++kz) {
                        for (int ky = 0; ky < n_qp; ++ky) {
                            for (int kx = 0; kx < n_qp; ++kx) {
                                hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], Jadj, &Jdet);
                                hex8_neohookean_ogden_grad(Jadj,
                                                           Jdet,
                                                           qx[kx],
                                                           qx[ky],
                                                           qx[kz],
                                                           qw[kx] * qw[ky] * qw[kz],
                                                           mu,
                                                           lambda,
                                                           ux,
                                                           uy,
                                                           uz,
                                                           eoutx,
                                                           eouty,
                                                           eoutz);
                            }
                        }
                    }
                };

                Hyperelasticity3 helper;
                return helper.gradient<8>(micro_kernel,
                                          nelements,
                                          /*stride*/ 1,
                                          nnodes,
                                          elements,
                                          points,
                                          /*u_stride*/ 3,
                                          &u[0],
                                          &u[1],
                                          &u[2],
                                          /*out_stride*/ 3,
                                          &out[0],
                                          &out[1],
                                          &out[2]);
            }

            int apply_aos(const ptrdiff_t      nelements,
                          const ptrdiff_t      nnodes,
                          idx_t **const        elements,
                          geom_t **const       points,
                          const real_t *const  u,
                          const real_t *const  h,
                          real_t *const        out) override {
                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                auto s_kernel = [mu, lambda](const scalar_t *lx,
                                             const scalar_t *ly,
                                             const scalar_t *lz,
                                             const scalar_t *ux,
                                             const scalar_t *uy,
                                             const scalar_t *uz,
                                             scalar_t *const S_out) {
                    static const scalar_t sample = 0.5;
                    scalar_t               Jadj[9];
                    scalar_t               Jdet;
                    hex8_adjugate_and_det(lx, ly, lz, sample, sample, sample, Jadj, &Jdet);
                    scalar_t F[9] = {0};
                    hex8_F(Jadj, Jdet, sample, sample, sample, ux, uy, uz, F);
                    scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
                    hex8_S_ikmn_neohookean(Jadj, Jdet, sample, sample, sample, 1, F, mu, lambda, S_ikmn);
                    for (int k = 0; k < HEX8_S_IKMN_SIZE; ++k) S_out[k] = S_ikmn[k];
                };

                Hyperelasticity3 helper;
                return helper.apply_aos<8, 10, HEX8_S_IKMN_SIZE>(s_kernel,
                                                                 hex8_Wimpn_compressed,
                                                                 hex8_SdotHdotG,
                                                                 nelements,
                                                                 /*stride*/ 1,
                                                                 elements,
                                                                 points,
                                                                 /*u_stride*/ 3,
                                                                 &u[0],
                                                                 &u[1],
                                                                 &u[2],
                                                                 /*h_stride*/ 3,
                                                                 &h[0],
                                                                 &h[1],
                                                                 &h[2],
                                                                 /*out_stride*/ 3,
                                                                 &out[0],
                                                                 &out[1],
                                                                 &out[2]);
            }

            int partial_assembly_apply(const ptrdiff_t                      nelements,
                                       const ptrdiff_t                      stride,
                                       idx_t **const                        elements,
                                       const metric_tensor_t *const         partial_assembly,
                                       const ptrdiff_t                      h_stride,
                                       const real_t *const                  hx,
                                       const real_t *const                  hy,
                                       const real_t *const                  hz,
                                       const ptrdiff_t                      out_stride,
                                       real_t *const                        outx,
                                       real_t *const                        outy,
                                       real_t *const                        outz) override {
                Hyperelasticity3 helper;
                return helper.partial_assembly_apply<8, 10, HEX8_S_IKMN_SIZE>(
                        hex8_Wimpn_compressed,
                        hex8_SdotHdotG,
                        nelements,
                        stride,
                        elements,
                        partial_assembly,
                        h_stride,
                        hx,
                        hy,
                        hz,
                        out_stride,
                        outx,
                        outy,
                        outz);
            }

            int compressed_partial_assembly_apply(const ptrdiff_t                nelements,
                                                  const ptrdiff_t                stride,
                                                  idx_t **const                  elements,
                                                  const compressed_t *const      partial_assembly,
                                                  const scaling_t *const         scaling,
                                                  const ptrdiff_t                h_stride,
                                                  const real_t *const            hx,
                                                  const real_t *const            hy,
                                                  const real_t *const            hz,
                                                  const ptrdiff_t                out_stride,
                                                  real_t *const                  outx,
                                                  real_t *const                  outy,
                                                  real_t *const                  outz) override {
                Hyperelasticity3 helper;
                return helper.compressed_partial_assembly_apply<8, 10, HEX8_S_IKMN_SIZE>(
                        hex8_Wimpn_compressed,
                        hex8_SdotHdotG,
                        nelements,
                        stride,
                        elements,
                        partial_assembly,
                        scaling,
                        h_stride,
                        hx,
                        hy,
                        hz,
                        out_stride,
                        outx,
                        outy,
                        outz);
            }

            int hessian_partial_assembly(const ptrdiff_t         nelements,
                                         const ptrdiff_t         stride,
                                         idx_t **const           elements,
                                         geom_t **const          points,
                                         const ptrdiff_t         u_stride,
                                         const real_t *const     ux,
                                         const real_t *const     uy,
                                         const real_t *const     uz,
                                         metric_tensor_t *const  pa) override {

                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                return neohookean_ogden_hessian_partial_assembly(
                        HEX8, nelements, stride, elements, points, mu, lambda, u_stride, ux, uy, uz, pa);
            }

            // Objective and path-evaluated objective (HEX8 specialized forms)
            int objective(const ptrdiff_t    nelements,
                          const ptrdiff_t    /*stride*/,  // elements assumed SoA with stride 1
                          const ptrdiff_t    nnodes,
                          idx_t **const      elements,
                          geom_t **const     points,
                          const ptrdiff_t    u_stride,
                          const real_t *const ux,
                          const real_t *const uy,
                          const real_t *const uz,
                          const int          is_elem_wise,
                          real_t *const      out) override {

                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                auto micro_kernel = [mu, lambda](const scalar_t *lx,
                                                 const scalar_t *ly,
                                                 const scalar_t *lz,
                                                 const scalar_t *ux,
                                                 const scalar_t *uy,
                                                 const scalar_t *uz,
                                                 scalar_t *      value) {
                    static const int       n_qp = line_q2_n;
                    static const scalar_t *qx   = line_q2_x;
                    static const scalar_t *qw   = line_q2_w;
                    hex8_neohookean_ogden_objective_integral(lx, ly, lz, n_qp, qx, qw, mu, lambda, ux, uy, uz, value);
                };

                return Hyperelasticity3::objective<8>(micro_kernel,
                                                      nelements,
                                                      /*stride*/ 1,
                                                      nnodes,
                                                      elements,
                                                      points,
                                                      u_stride,
                                                      ux,
                                                      uy,
                                                      uz,
                                                      is_elem_wise,
                                                      out);
            }

            int objective_steps(const ptrdiff_t    nelements,
                                const ptrdiff_t    /*stride*/,  // elements assumed SoA with stride 1
                                const ptrdiff_t    nnodes,
                                idx_t **const      elements,
                                geom_t **const     points,
                                const ptrdiff_t    u_stride,
                                const real_t *const ux,
                                const real_t *const uy,
                                const real_t *const uz,
                                const ptrdiff_t    inc_stride,
                                const real_t *const incx,
                                const real_t *const incy,
                                const real_t *const incz,
                                const int          nsteps,
                                const real_t *const steps,
                                real_t *const      out) override {

                const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
                const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

                auto micro_kernel = [mu, lambda](const scalar_t *lx,
                                                 const scalar_t *ly,
                                                 const scalar_t *lz,
                                                 const scalar_t *ux,
                                                 const scalar_t *uy,
                                                 const scalar_t *uz,
                                                 const scalar_t *incx,
                                                 const scalar_t *incy,
                                                 const scalar_t *incz,
                                                 const int       nsteps,
                                                 const scalar_t *steps,
                                                 scalar_t *      out_local) {
                    static const int       n_qp = line_q2_n;
                    static const scalar_t *qx   = line_q2_x;
                    static const scalar_t *qw   = line_q2_w;
                    hex8_neohookean_ogden_objective_steps_integral(
                            lx, ly, lz, n_qp, qx, qw, mu, lambda, ux, uy, uz, incx, incy, incz, nsteps, steps, out_local);
                };

                Hyperelasticity3 helper;
                return helper.objective_steps<8>(micro_kernel,
                                                 nelements,
                                                 /*stride*/ 1,
                                                 nnodes,
                                                 elements,
                                                 points,
                                                 mu,
                                                 lambda,
                                                 u_stride,
                                                 ux,
                                                 uy,
                                                 uz,
                                                 inc_stride,
                                                 incx,
                                                 incy,
                                                 incz,
                                                 nsteps,
                                                 steps,
                                                 out);
            }
        };

        class MooneyRivlin final : public HyperelasticityKernels {
        public:
            int hessian_aos(const ptrdiff_t,
                            const ptrdiff_t,
                            idx_t **const,
                            geom_t **const,
                            const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) override {
                return SFEM_SUCCESS;
            }

            int hessian_diag_aos(const ptrdiff_t,
                                 const ptrdiff_t,
                                 idx_t **const,
                                 geom_t **const,
                                 const real_t *const,
                                 real_t *const) override {
                return SFEM_SUCCESS;
            }

            int gradient_aos(const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, const real_t *const, real_t *const)
                    override {
                return SFEM_SUCCESS;
            }

            int apply_aos(const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const real_t *const,
                          const real_t *const,
                          real_t *const) override {
                return SFEM_SUCCESS;
            }

            int partial_assembly_apply(const ptrdiff_t,
                                       const ptrdiff_t,
                                       idx_t **const,
                                       const metric_tensor_t *const,
                                       const ptrdiff_t,
                                       const real_t *const,
                                       const real_t *const,
                                       const real_t *const,
                                       const ptrdiff_t,
                                       real_t *const,
                                       real_t *const,
                                       real_t *const) override {
                return SFEM_SUCCESS;
            }

            int compressed_partial_assembly_apply(const ptrdiff_t,
                                                  const ptrdiff_t,
                                                  idx_t **const,
                                                  const compressed_t *const,
                                                  const scaling_t *const,
                                                  const ptrdiff_t,
                                                  const real_t *const,
                                                  const real_t *const,
                                                  const real_t *const,
                                                  const ptrdiff_t,
                                                  real_t *const,
                                                  real_t *const,
                                                  real_t *const) override {
                return SFEM_SUCCESS;
            }

            int hessian_partial_assembly(const ptrdiff_t,
                                         const ptrdiff_t,
                                         idx_t **const,
                                         geom_t **const,
                                         const ptrdiff_t,
                                         const real_t *const,
                                         const real_t *const,
                                         const real_t *const,
                                         metric_tensor_t *const) override {
                return SFEM_SUCCESS;
            }

            // Objective and path-evaluated objective (HEX8 specialized forms)
            int objective(const ptrdiff_t,
                          const ptrdiff_t,
                          const ptrdiff_t,
                          idx_t **const,
                          geom_t **const,
                          const ptrdiff_t,
                          const real_t *const,
                          const real_t *const,
                          const real_t *const,
                          const int,
                          real_t *const) override {
                return SFEM_SUCCESS;
            }

            int objective_steps(const ptrdiff_t,
                                const ptrdiff_t,
                                const ptrdiff_t,
                                idx_t **const,
                                geom_t **const,
                                const ptrdiff_t,
                                const real_t *const,
                                const real_t *const,
                                const real_t *const,
                                const ptrdiff_t,
                                const real_t *const,
                                const real_t *const,
                                const real_t *const,
                                const int,
                                const real_t *const,
                                real_t *const) override {
                return SFEM_SUCCESS;
            }
        };
    }  // namespace hex8

    class Hyperelasticity::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }

        std::unordered_map<enum ElemType, std::shared_ptr<HyperelasticityKernels>> kernels;
        std::shared_ptr<HyperelasticityKernels> find_kernels(const OpDomain &domain) { return kernels[domain.element_type]; }
    };

    std::unique_ptr<Op> Hyperelasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
        auto ret           = std::make_unique<Hyperelasticity>(space);
        // Register HEX8 kernels
        ret->impl_->kernels[HEX8] = std::make_shared<hex8::NeoHookeanOgden>();
        // auto plugin_folder = sfem::Env::read_string("SFEM_HYPERELASTICITY_PLUGIN_FOLDER", "./");
        // if (!plugin_folder.empty()) {
        //     ret->impl_->hyperelasticity_load_plugins(plugin_folder);
        // }
        return ret;
    }

    Hyperelasticity::Hyperelasticity(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    int Hyperelasticity::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("Hyperelasticity::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->node_to_node_graph();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->hessian_aos(domain.block->n_elements(),
                                        mesh->n_nodes(),
                                        domain.block->elements()->data(),
                                        mesh->points()->data(),
                                        x,
                                        graph->rowptr()->data(),
                                        graph->colidx()->data(),
                                        values);
        });
    }

    int Hyperelasticity::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->hessian_diag_aos(
                    mesh->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), mesh->points()->data(), x, out);
        });
    }

    int Hyperelasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto kernels = impl_->find_kernels(domain);
            return kernels->gradient_aos(domain.block->n_elements(),
                                         mesh->n_nodes(),
                                         domain.block->elements()->data(),
                                         mesh->points()->data(),
                                         x,
                                         out);
        });
    }

    int Hyperelasticity::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::apply");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<struct HyperelasticityAssemblyData>(domain.user_data);
            if (ua->partial_assembly_buffer) {
                if (ua->use_compression) {
                    auto kernels = impl_->find_kernels(domain);
                    return kernels->compressed_partial_assembly_apply(domain.block->n_elements(),
                                                                      ua->elements_stride,
                                                                      ua->elements->data(),
                                                                      ua->partial_assembly_compressed->data(),
                                                                      ua->compression_scaling->data(),
                                                                      3,
                                                                      &h[0],
                                                                      &h[1],
                                                                      &h[2],
                                                                      3,
                                                                      &out[0],
                                                                      &out[1],
                                                                      &out[2]);

                } else {
                    auto kernels = impl_->find_kernels(domain);
                    return kernels->partial_assembly_apply(domain.block->n_elements(),
                                                           ua->elements_stride,
                                                           ua->elements->data(),
                                                           ua->partial_assembly_buffer->data(),
                                                           3,
                                                           &h[0],
                                                           &h[1],
                                                           &h[2],
                                                           3,
                                                           &out[0],
                                                           &out[1],
                                                           &out[2]);
                }
            }

            auto kernels = impl_->find_kernels(domain);
            return kernels->apply_aos(domain.block->n_elements(),
                                      mesh->n_nodes(),
                                      domain.block->elements()->data(),
                                      mesh->points()->data(),
                                      x,
                                      h,
                                      out);
        });
    }

    int Hyperelasticity::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        bool use_partial_assembly = sfem::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", false);
        bool use_compression      = sfem::Env::read("SFEM_USE_COMPRESSION", false);
        bool use_AoS              = sfem::Env::read("SFEM_HYPERELAStICITY_USE_AOS", false);

        for (auto &domain : impl_->domains->domains()) {
            auto ua                  = std::make_shared<struct HyperelasticityAssemblyData>();
            ua->use_partial_assembly = use_partial_assembly || domain.second.element_type == HEX8;
            ua->use_compression      = use_compression;
            ua->use_AoS              = use_AoS;
            ua->elements             = domain.second.block->elements();
            ua->elements_stride      = 1;
            domain.second.user_data  = ua;

            if (use_AoS) {
                auto nxe            = domain.second.block->n_nodes_per_element();
                ua->elements        = convert_host_buffer_to_fake_SoA(nxe, soa_to_aos(1, nxe, domain.second.block->elements()));
                ua->elements_stride = nxe;
            }
        }

        return SFEM_SUCCESS;
    }

    int Hyperelasticity::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("Hyperelasticity::update");

        auto mesh = impl_->space->mesh_ptr();

        for (auto &domain : impl_->domains->domains()) {
            auto assembly_data = std::static_pointer_cast<struct HyperelasticityAssemblyData>(domain.second.user_data);
            if (!assembly_data->use_partial_assembly) continue;

            auto element_type = domain.second.element_type;

            if (element_type == TET4 || element_type == HEX8) {
                // FIXME: Add support for other element types
                if (!assembly_data->partial_assembly_buffer) {
                    assembly_data->partial_assembly_buffer =
                            sfem::create_host_buffer<metric_tensor_t>(domain.second.block->n_elements() * IKMN_SIZE);
                }
                auto kernels = impl_->find_kernels(domain.second);
                int  ok      = kernels->hessian_partial_assembly(domain.second.block->n_elements(),
                                                           assembly_data->elements_stride,
                                                           assembly_data->elements->data(),
                                                           mesh->points()->data(),
                                                           3,
                                                           &u[0],
                                                           &u[1],
                                                           &u[2],
                                                           assembly_data->partial_assembly_buffer->data());
                assembly_data->compress_partial_assembly(domain.second);
            }
        }

        return SFEM_SUCCESS;
    }

    int Hyperelasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<HyperelasticityAssemblyData>(domain.user_data);

            auto kernels = impl_->find_kernels(domain);
            return kernels->objective(domain.block->n_elements(),
                                      ua->elements_stride,
                                      mesh->n_nodes(),
                                      ua->elements->data(),
                                      mesh->points()->data(),
                                      3,
                                      &x[0],
                                      &x[1],
                                      &x[2],
                                      false,
                                      out);
        });
    }

    int Hyperelasticity::value_steps(const real_t       *x,
                                     const real_t       *h,
                                     const int           nsteps,
                                     const real_t *const steps,
                                     real_t *const       out) {
        SFEM_TRACE_SCOPE("Hyperelasticity::value_steps");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<HyperelasticityAssemblyData>(domain.user_data);
            if (domain.element_type == HEX8) {
                auto kernels = impl_->find_kernels(domain);
                return kernels->objective_steps(mesh->n_elements(),
                                                ua->elements_stride,
                                                mesh->n_nodes(),
                                                ua->elements->data(),
                                                mesh->points()->data(),
                                                3,
                                                &x[0],
                                                &x[1],
                                                &x[2],
                                                3,
                                                &h[0],
                                                &h[1],
                                                &h[2],
                                                nsteps,
                                                steps,
                                                out);
            } else {
                // Must be implemented
                return SFEM_FAILURE;
            }
        });
    }

    int Hyperelasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Hyperelasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::lor_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<Hyperelasticity>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Hyperelasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::derefine_op");

        // FIXME: Must work for all element types and multi-block
        auto ret            = std::make_shared<Hyperelasticity>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Hyperelasticity::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        auto ret = std::make_shared<Hyperelasticity>(impl_->space);
        return ret;
    }

    Hyperelasticity::~Hyperelasticity() = default;

    void Hyperelasticity::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Hyperelasticity::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem
