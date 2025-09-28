#include "sfem_Hyperelasticity.hpp"

#include "neohookean_ogden.h"
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

// FIXME
#include "hex8_neohookean_ogden.h"
#include "tet4_neohookean_ogden.h"
#include "tet4_partial_assembly_neohookean_inline.h"

#include <dlfcn.h>
#include <math.h>
#include <mpi.h>
#include <functional>
#include <vector>

namespace sfem {

    struct HyperelasticityAssemblyData {
        SharedBuffer<metric_tensor_t> partial_assembly_buffer;
        SharedBuffer<scaling_t>       compression_scaling;
        SharedBuffer<compressed_t>    partial_assembly_compressed;
        SharedBuffer<idx_t *>         elements;
        ptrdiff_t                     elements_stride{1};

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};

        int compress_partial_assembly(OpDomain &domain) {
            auto mesh = domain.block;

            if (use_compression) {
                if (!compression_scaling) {
                    compression_scaling         = sfem::create_host_buffer<scaling_t>(mesh->n_elements());
                    partial_assembly_compressed = sfem::create_host_buffer<compressed_t>(mesh->n_elements() * TET4_S_IKMN_SIZE);
                }

                auto      cs         = compression_scaling->data();
                auto      pa         = partial_assembly_buffer->data();
                auto      pac        = partial_assembly_compressed->data();
                ptrdiff_t n_elements = mesh->n_elements();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_elements; i++) {
                    auto pai = &pa[i * TET4_S_IKMN_SIZE];
                    cs[i]    = pai[0];
                    for (int v = 1; v < TET4_S_IKMN_SIZE; v++) {
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
                    auto pai  = &pa[i * TET4_S_IKMN_SIZE];
                    auto paci = &pac[i * TET4_S_IKMN_SIZE];
                    for (int v = 0; v < TET4_S_IKMN_SIZE; v++) {
                        paci[v] = (compressed_t)(pai[v] / cs[i]);

                        assert(cs[i] > 0);
                        assert(std::isfinite(paci[v]));
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

    class Hyperelasticity::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }

        std::unordered_map<enum ElemType, std::shared_ptr<HyperelasticityKernels>> kernels;

        std::shared_ptr<HyperelasticityKernels> find_kernels(const OpDomain &domain) { return kernels[domain.element_type]; }

        int hyperelasticity_load_plugins(const std::string &folder) {
            // Load candidate shared libraries from folder (best‑effort),
            // resolve known hyperelasticity symbols, and bind fast wrappers.

            auto try_open_all = [&](const std::string &dir) -> std::vector<void *> {
                std::vector<void *> handles;
                auto                so_files    = sfem::find_files(dir + "/*.so");
                auto                dylib_files = sfem::find_files(dir + "/*.dylib");
                so_files.insert(so_files.end(), dylib_files.begin(), dylib_files.end());
                for (const auto &path : so_files) {
                    void *h = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
                    if (h) handles.push_back(h);
                }
                return handles;
            };

            auto handles = try_open_all(folder);

            // Helper: resolve symbol from any opened handle
            auto resolve = [&](const char *sym) -> void * {
                for (auto *h : handles) {
                    if (!h) continue;
                    void *p = dlsym(h, sym);
                    if (p) return p;
                }
                return nullptr;
            };

            // // Typedefs for underlying C kernels
            // using hess_aos_fn = int (*)(void *ctx,
            //                             const enum ElemType,
            //                             const ptrdiff_t,
            //                             const ptrdiff_t,
            //                             idx_t **const,
            //                             geom_t **const,
            //                             const real_t,
            //                             const real_t,
            //                             const real_t *const,
            //                             const count_t *const,
            //                             const idx_t *const,
            //                             real_t *const);

            // using hess_diag_aos_fn = int (*)(void *ctx,
            //                                  const enum ElemType,
            //                                  const ptrdiff_t,
            //                                  const ptrdiff_t,
            //                                  idx_t **const,
            //                                  geom_t **const,
            //                                  const real_t,
            //                                  const real_t,
            //                                  const real_t *const,
            //                                  real_t *const);

            // using grad_aos_fn = int (*)(void *ctx,
            //                             const enum ElemType,
            //                             const ptrdiff_t,
            //                             const ptrdiff_t,
            //                             idx_t **const,
            //                             geom_t **const,
            //                             const real_t,
            //                             const real_t,
            //                             const real_t *const,
            //                             real_t *const);

            // using apply_aos_fn = int (*)(void *ctx,
            //                              const enum ElemType,
            //                              const ptrdiff_t,
            //                              const ptrdiff_t,
            //                              idx_t **const,
            //                              geom_t **const,
            //                              const real_t,
            //                              const real_t,
            //                              const real_t *const,
            //                              const real_t *const,
            //                              real_t *const);

            // using hess_pa_fn = int (*)(void *ctx,
            //                            const enum ElemType,
            //                            const ptrdiff_t,
            //                            const ptrdiff_t,
            //                            idx_t **const,
            //                            geom_t **const,
            //                            const real_t,
            //                            const real_t,
            //                            const ptrdiff_t,
            //                            const real_t *const,
            //                            const real_t *const,
            //                            const real_t *const,
            //                            metric_tensor_t *const);

            // using pa_apply_fn = int (*)(void *ctx,
            //                             const enum ElemType,
            //                             const ptrdiff_t,
            //                             const ptrdiff_t,
            //                             idx_t **const,
            //                             const metric_tensor_t *const,
            //                             const ptrdiff_t,
            //                             const real_t *const,
            //                             const real_t *const,
            //                             const real_t *const,
            //                             const ptrdiff_t,
            //                             real_t *const,
            //                             real_t *const,
            //                             real_t *const);

            // using pa_apply_comp_fn = int (*)(void *ctx,
            //                                  const enum ElemType,
            //                                  const ptrdiff_t,
            //                                  const ptrdiff_t,
            //                                  idx_t **const,
            //                                  const compressed_t *const,
            //                                  const scaling_t *const,
            //                                  const ptrdiff_t,
            //                                  const real_t *const,
            //                                  const real_t *const,
            //                                  const real_t *const,
            //                                  const ptrdiff_t,
            //                                  real_t *const,
            //                                  real_t *const,
            //                                  real_t *const);

            // using hex8_obj_fn = int (*)(void *ctx,
            //                             const ptrdiff_t,
            //                             const ptrdiff_t,
            //                             const ptrdiff_t,
            //                             idx_t **const,
            //                             geom_t **const,
            //                             const real_t,
            //                             const real_t,
            //                             const ptrdiff_t,
            //                             const real_t *const,
            //                             const real_t *const,
            //                             const real_t *const,
            //                             const int,
            //                             real_t *const);

            // using hex8_obj_steps_fn = int (*)(void *ctx,
            //                                   const ptrdiff_t,
            //                                   const ptrdiff_t,
            //                                   const ptrdiff_t,
            //                                   idx_t **const,
            //                                   geom_t **const,
            //                                   const real_t,
            //                                   const real_t,
            //                                   const ptrdiff_t,
            //                                   const real_t *const,
            //                                   const real_t *const,
            //                                   const real_t *const,
            //                                   const ptrdiff_t,
            //                                   const real_t *const,
            //                                   const real_t *const,
            //                                   const real_t *const,
            //                                   const int,
            //                                   const real_t *const,
            //                                   real_t *const);

            // // Resolve (prefer plug-ins; otherwise use built-ins)
            // auto f_hess_aos = reinterpret_cast<hess_aos_fn>(resolve("neohookean_ogden_hessian_aos"));
            // if (!f_hess_aos) f_hess_aos = &neohookean_ogden_hessian_aos;

            // auto f_hess_diag_aos = reinterpret_cast<hess_diag_aos_fn>(resolve("neohookean_ogden_diag_aos"));
            // if (!f_hess_diag_aos) f_hess_diag_aos = &neohookean_ogden_diag_aos;

            // auto f_grad_aos = reinterpret_cast<grad_aos_fn>(resolve("neohookean_ogden_gradient_aos"));
            // if (!f_grad_aos) f_grad_aos = &neohookean_ogden_gradient_aos;

            // auto f_apply_aos = reinterpret_cast<apply_aos_fn>(resolve("neohookean_ogden_apply_aos"));
            // if (!f_apply_aos) f_apply_aos = &neohookean_ogden_apply_aos;

            // auto f_hess_pa = reinterpret_cast<hess_pa_fn>(resolve("neohookean_ogden_hessian_partial_assembly"));
            // if (!f_hess_pa) f_hess_pa = &neohookean_ogden_hessian_partial_assembly;

            // auto f_pa_apply = reinterpret_cast<pa_apply_fn>(resolve("neohookean_ogden_partial_assembly_apply"));
            // if (!f_pa_apply) f_pa_apply = &neohookean_ogden_partial_assembly_apply;

            // auto f_pa_apply_comp =
            //         reinterpret_cast<pa_apply_comp_fn>(resolve("neohookean_ogden_compressed_partial_assembly_apply"));
            // if (!f_pa_apply_comp) f_pa_apply_comp = &neohookean_ogden_compressed_partial_assembly_apply;

            // auto f_hex8_obj = reinterpret_cast<hex8_obj_fn>(resolve("hex8_neohookean_ogden_objective"));
            // if (!f_hex8_obj) f_hex8_obj = &hex8_neohookean_ogden_objective;

            // auto f_hex8_obj_steps = reinterpret_cast<hex8_obj_steps_fn>(resolve("hex8_neohookean_ogden_objective_steps"));
            // if (!f_hex8_obj_steps) f_hex8_obj_steps = &hex8_neohookean_ogden_objective_steps;

            // // Bind mu, lambda once (fast path; per-block overrides are not supported here)
            // // const real_t mu     = sfem::Env::read("SFEM_SHEAR_MODULUS", real_t(1));
            // // const real_t lambda = sfem::Env::read("SFEM_FIRST_LAME_PARAMETER", real_t(1));

            // auto make_common = [&]() {
            //     auto k = std::make_shared<HyperelasticityKernels>();

            //     k->hessian_aos = [=](const enum ElemType  et,
            //                          const ptrdiff_t      ne,
            //                          const ptrdiff_t      nn,
            //                          idx_t **const        els,
            //                          geom_t **const       pts,
            //                          const real_t *const  u,
            //                          const count_t *const rowptr,
            //                          const idx_t *const   colidx,
            //                          real_t *const        values) {
            //         return f_hess_aos(ctx, et, ne, nn, els, pts, u, rowptr, colidx, values);
            //     };

            //     k->hessian_diag_aos = [=](const enum ElemType et,
            //                               const ptrdiff_t     ne,
            //                               const ptrdiff_t     nn,
            //                               idx_t **const       els,
            //                               geom_t **const      pts,
            //                               const real_t *const u,
            //                               real_t *const       out) { return f_hess_diag_aos(ctx, et, ne, nn, els, pts, u, out);
            //                               };

            //     k->gradient_aos = [=](const enum ElemType et,
            //                           const ptrdiff_t     ne,
            //                           const ptrdiff_t     nn,
            //                           idx_t **const       els,
            //                           geom_t **const      pts,
            //                           const real_t *const u,
            //                           real_t *const       out) { return f_grad_aos(ctx, et, ne, nn, els, pts, u, out); };

            //     k->apply_aos = [=](const enum ElemType et,
            //                        const ptrdiff_t     ne,
            //                        const ptrdiff_t     nn,
            //                        idx_t **const       els,
            //                        geom_t **const      pts,
            //                        const real_t *const x,
            //                        const real_t *const h,
            //                        real_t *const       out) { return f_apply_aos(ctx, et, ne, nn, els, pts, x, h, out); };

            //     k->partial_assembly_apply = [=](const enum ElemType          et,
            //                                     const ptrdiff_t              ne,
            //                                     const ptrdiff_t              stride,
            //                                     idx_t **const                els,
            //                                     const metric_tensor_t *const partial,
            //                                     const ptrdiff_t              h_stride,
            //                                     const real_t *const          hx,
            //                                     const real_t *const          hy,
            //                                     const real_t *const          hz,
            //                                     const ptrdiff_t              out_stride,
            //                                     real_t *const                outx,
            //                                     real_t *const                outy,
            //                                     real_t *const                outz) {
            //         return f_pa_apply(ctx, et, ne, stride, els, partial, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
            //     };

            //     k->compressed_partial_assembly_apply = [=](const enum ElemType       et,
            //                                                const ptrdiff_t           ne,
            //                                                const ptrdiff_t           stride,
            //                                                idx_t **const             els,
            //                                                const compressed_t *const partial,
            //                                                const scaling_t *const    scaling,
            //                                                const ptrdiff_t           h_stride,
            //                                                const real_t *const       hx,
            //                                                const real_t *const       hy,
            //                                                const real_t *const       hz,
            //                                                const ptrdiff_t           out_stride,
            //                                                real_t *const             outx,
            //                                                real_t *const             outy,
            //                                                real_t *const             outz) {
            //         return f_pa_apply_comp(
            //                 ctx, et, ne, stride, els, partial, scaling, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
            //     };

            //     k->hessian_partial_assembly = [=](const enum ElemType    et,
            //                                       const ptrdiff_t        ne,
            //                                       const ptrdiff_t        stride,
            //                                       idx_t **const          els,
            //                                       geom_t **const         pts,
            //                                       const ptrdiff_t        u_stride,
            //                                       const real_t *const    ux,
            //                                       const real_t *const    uy,
            //                                       const real_t *const    uz,
            //                                       metric_tensor_t *const partial) {
            //         return f_hess_pa(ctx, et, ne, stride, els, pts, u_stride, ux, uy, uz, partial);
            //     };

            //     return k;
            // };

            // // Common kernels for all supported element types
            // auto common_tet4 = make_common();
            // auto common_hex8 = make_common();

            // // HEX8 objective kernels
            // common_hex8->objective = [=](const ptrdiff_t     ne,
            //                              const ptrdiff_t     stride,
            //                              const ptrdiff_t     nn,
            //                              idx_t **const       els,
            //                              geom_t **const      pts,
            //                              const ptrdiff_t     u_stride,
            //                              const real_t *const ux,
            //                              const real_t *const uy,
            //                              const real_t *const uz,
            //                              const int           is_element_wise,
            //                              real_t *const       out) {
            //     return f_hex8_obj(ctx, ne, stride, nn, els, pts, u_stride, ux, uy, uz, is_element_wise, out);
            // };

            // common_hex8->objective_steps = [=](const ptrdiff_t     ne,
            //                                    const ptrdiff_t     stride,
            //                                    const ptrdiff_t     nn,
            //                                    idx_t **const       els,
            //                                    geom_t **const      pts,
            //                                    const ptrdiff_t     u_stride,
            //                                    const real_t *const ux,
            //                                    const real_t *const uy,
            //                                    const real_t *const uz,
            //                                    const ptrdiff_t     inc_stride,
            //                                    const real_t *const incx,
            //                                    const real_t *const incy,
            //                                    const real_t *const incz,
            //                                    const int           nsteps,
            //                                    const real_t *const steps,
            //                                    real_t *const       out) {
            //     return f_hex8_obj_steps(ctx,
            //                             ne,
            //                             stride,
            //                             nn,
            //                             els,
            //                             pts,
            //                             ctx,
            //                             u_stride,
            //                             ux,
            //                             uy,
            //                             uz,
            //                             inc_stride,
            //                             incx,
            //                             incy,
            //                             incz,
            //                             nsteps,
            //                             steps,
            //                             out);
            // };

            // // TET4 has no objective in this fast path
            // common_tet4->objective       = [](auto...) { return SFEM_FAILURE; };
            // common_tet4->objective_steps = [](auto...) { return SFEM_FAILURE; };

            // // Register per-element kernels
            // kernels[TET4] = common_tet4;
            // kernels[HEX8] = common_hex8;

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> Hyperelasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Hyperelasticity::create");

        assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
        auto ret           = std::make_unique<Hyperelasticity>(space);
        auto plugin_folder = sfem::Env::read_string("SFEM_HYPERELASTICITY_PLUGIN_FOLDER", "./");
        if (!plugin_folder.empty()) {
            ret->impl_->hyperelasticity_load_plugins(plugin_folder);
        }
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
                            sfem::create_host_buffer<metric_tensor_t>(domain.second.block->n_elements() * TET4_S_IKMN_SIZE);
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
