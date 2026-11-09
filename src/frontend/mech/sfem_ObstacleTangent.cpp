#include "sfem_ObstacleTangent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_InexactInterface.hpp"
#include "sfem_LinearOperatorFactory.hpp"
#include "sfem_MemoryLedger.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_TeamBLAS.hpp"
#include "sfem_blas_factory.hpp"
#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#endif

namespace sfem {

    namespace {

        class AssembledTangent final : public ObstacleTangent {
        public:
            std::shared_ptr<BSRTangent>                   tangent;
            std::shared_ptr<NodeObstacleContact>          contact;
            std::shared_ptr<Operator<real_t>>             product;
            std::shared_ptr<TeamOperator<real_t>> team_product;
            // The curvature the values carry (set_curvature), to take back before another.
            bool curved{false};
            int  curvature{0};

            int assemble(const real_t *const x) override {
                if (tangent->assemble(x) != SFEM_SUCCESS) return SFEM_FAILURE;
                curved = false;
                // y = K x: the QP's product writes its output whole.
                product = tangent->op(0);
                if (tangent->execution_space() == EXECUTION_SPACE_HOST) {
                    team_product = std::dynamic_pointer_cast<TeamOperator<real_t>>(product);
                    if (!team_product) SFEM_ERROR("ObstacleTangent: the host tangent has no product for a region of its caller\n");
                }
                return SFEM_SUCCESS;
            }

            int set_curvature(const real_t *const x, const real_t *const r, const real_t *const k, const int curvature_now) override {
                if (curved) add(x, r, k, curvature, -1);
                add(x, r, k, curvature_now, 1);
                curved    = true;
                curvature = curvature_now;
                return SFEM_SUCCESS;
            }

            int shift_diagonal(const real_t factor, const real_t *const reference) override {
                return tangent->shift_diagonal(factor, reference);
            }

            int diagonal(real_t *const out) override { return tangent->diagonal(out); }

            real_t max_eig(const real_t *const scale) const override { return tangent->max_abs_row_sum(scale); }

            std::shared_ptr<Operator<real_t>>             op() const override { return product; }
            std::shared_ptr<TeamOperator<real_t>> team() const override { return team_product; }
            bool team_capable() const override { return tangent->execution_space() == EXECUTION_SPACE_HOST; }

            void account(MemoryLedger &ledger) const override { tangent->account(ledger); }

        private:
            void add(const real_t *const x, const real_t *const r, const real_t *const k, const int curvature_term, const real_t sign) {
                contact->add_curvature(
                        x, r, k, curvature_term, sign, tangent->rowptr(), tangent->colidx(), tangent->values_type(), tangent->values());
            }
        };

        class ComposedTangent final : public ObstacleTangent {
        public:
            std::shared_ptr<Function>            f;
            std::shared_ptr<NodeObstacleContact> contact;
            std::string                          format;
            ExecutionSpace                       es{EXECUTION_SPACE_HOST};
            ptrdiff_t                            ndofs{0};
            std::shared_ptr<BLAS<real_t>>        blas;
            // The state the material is linearised at (MF reads it at each product).
            SharedBuffer<real_t>                 state;
            std::shared_ptr<Operator<real_t>>    material;
            std::shared_ptr<Operator<real_t>>    product;
            // The product on a region's threads, where the material has one (host).
            std::shared_ptr<TeamOperator<real_t>> team_product;
            // 6 per contact node (node_obstacle_curvature_blocks), when `curved`.
            SharedBuffer<real_t>                 blocks;
            bool                                 curved{false};
            // K_dd += shift reference_d (shift_diagonal), with tmp for its product.
            SharedBuffer<real_t>                 reference, tmp;
            real_t                               shift{0};
            // Where the stored tangent takes the curvature and the shift in its own pass
            // (Function::set_inexact_extra, the device's assembled store): the extra the product
            // hands it.
            bool                                 fused{false};
            inexact::DiagonalExtra               extra;

            int assemble(const real_t *const x) override {
                blas->copy(ndofs, x, state->data());
                if (format == op_type::INEXACT && f->inexact_supported() && f->inexact_update(x) != SFEM_SUCCESS) {
                    return SFEM_FAILURE;
                }
                curved = false;
                shift  = 0;
                return SFEM_SUCCESS;
            }

            int set_curvature(const real_t *const x, const real_t *const r, const real_t *const k, const int curvature) override {
                contact->curvature_blocks(x, r, k, curvature, blocks->data());
                curved = true;
                return SFEM_SUCCESS;
            }

            int shift_diagonal(const real_t factor, const real_t *const d) override {
                if (shift == 0) absolute_value(d, reference->data());
                shift += factor;
                return SFEM_SUCCESS;
            }

            int diagonal(real_t *const out) override {
                blas->zeros(ndofs, out);
                if (format == op_type::INEXACT && f->inexact_supported()) {
                    // The constrained rows are 1 (the identity).
                    if (f->inexact_diag(state->data(), out) != SFEM_SUCCESS) return SFEM_FAILURE;
                } else {
                    if (f->hessian_diag(state->data(), out) != SFEM_SUCCESS) {
                        fprintf(stderr,
                                "ObstacleTangent: format %s needs Function::hessian_diag, which an operator here does not "
                                "provide; use INEXACT\n",
                                format.c_str());
                        return SFEM_FAILURE;
                    }
                    f->set_value_to_constrained_dofs(1, out);
                }
                if (curved) contact->blocks_diagonal(blocks->data(), out);
                if (shift != 0) blas->axpy(ndofs, shift, reference->data(), out);
                return SFEM_SUCCESS;
            }

            real_t max_eig(const real_t *const) const override { return 0; }

            std::shared_ptr<Operator<real_t>> op() const override { return product; }
            std::shared_ptr<TeamOperator<real_t>> team() const override { return team_product; }
            bool team_capable() const override { return team_product != nullptr; }

            void account(MemoryLedger &ledger) const override {
                // The stored tangent's codes, where its operators run (zero matrix-free).
                ledger.add(MemoryLedger::TANGENT_VALUES,
                           f->inexact_bytes(),
                           es == EXECUTION_SPACE_DEVICE ? smesh::MEMORY_SPACE_DEVICE : smesh::MEMORY_SPACE_HOST);
                for (auto *buffer : {&state, &reference, &tmp, &blocks}) ledger.add(MemoryLedger::TANGENT_WORK, *buffer);
            }

            void build() {
                // The material's product accumulates (Function::inexact_apply, Function::apply), its
                // constrained rows the identity: written over a zeroed output.
                material          = create_linear_operator(format, f, state, es);
                ComposedTangent *self = this;
                if (auto material_team = std::dynamic_pointer_cast<TeamOperator<real_t>>(material)) {
                    // Each thread zeroes its rows and, once all have (the material writes anyone's),
                    // adds the material's product; the curvature and the shift are its own rows.
                    // Distributed, the rows are the owned ones and the vectors span the local dofs.
                    auto            parallel = std::dynamic_pointer_cast<ParallelOperator<real_t>>(material);
                    const ptrdiff_t rows     = parallel ? parallel->rows() : ndofs;
                    team_product             = make_team_op<real_t>(
                            f->space()->mesh()->comm(),
                            rows,
                            rows,
                            ndofs,
                            ndofs,
                            false,
                            [self, material_team](const Team &team, const real_t *const in, real_t *const out) {
                                team_blas::zeros(team.begin, team.end, out);
                                team.barrier();
                                material_team->team_apply(team, in, out);
                                if (self->curved) self->contact->apply_blocks(team, self->blocks->data(), in, out);
                                if (self->shift != 0) {
                                    const real_t *const d = self->reference->data();
                                    for (ptrdiff_t i = team.begin; i < team.end; ++i) out[i] += self->shift * (d[i] * in[i]);
                                }
                            },
                            [material_team] { return material_team->prepare(); });
                    product = team_product_operator();
                    return;
                }
                fused = es == EXECUTION_SPACE_DEVICE && format == op_type::INEXACT && f->set_inexact_extra(&extra, true);
                f->set_inexact_extra(nullptr, false);
                product           = make_op<real_t>(
                        ndofs,
                        ndofs,
                        [self](const real_t *const in, real_t *const out) {
                            if (self->fused) {
                                // One pass writes out: the stored product, the curvature, the shift.
                                self->extra.contact_of = self->curved ? self->contact->frame_of() : nullptr;
                                self->extra.blocks     = self->blocks->data();
                                self->extra.shift      = self->shift;
                                self->extra.reference  = self->reference->data();
                                self->f->set_inexact_extra(&self->extra, true);
                                self->material->apply(in, out);
                                self->f->set_inexact_extra(nullptr, false);
                                return;
                            }
                            self->blas->zeros(self->ndofs, out);
                            self->material->apply(in, out);
                            if (self->curved) self->contact->apply_blocks(self->blocks->data(), in, out);
                            if (self->shift != 0) {
                                self->blas->xypaz(self->ndofs, self->reference->data(), in, 0, self->tmp->data());
                                self->blas->axpy(self->ndofs, self->shift, self->tmp->data(), out);
                            }
                        },
                        es);
            }

        private:
            std::shared_ptr<Operator<real_t>> team_product_operator() const {
                return std::dynamic_pointer_cast<Operator<real_t>>(team_product);
            }

            // out = |d|, wherever the vectors live: once per shifted solve.
            void absolute_value(const real_t *const d, real_t *const out) const {
                std::vector<real_t> host(ndofs);
#ifdef SFEM_ENABLE_CUDA
                if (es == EXECUTION_SPACE_DEVICE) {
                    copy_device_to_host(ndofs * sizeof(real_t), d, host.data());
                    for (auto &v : host) v = std::abs(v);
                    copy_host_to_device(ndofs * sizeof(real_t), host.data(), out);
                    return;
                }
#endif
                for (ptrdiff_t i = 0; i < ndofs; ++i) out[i] = std::abs(d[i]);
            }
        };

    }  // namespace

    std::shared_ptr<ObstacleTangent> ObstacleTangent::create(const std::shared_ptr<Function>            &f,
                                                             const std::shared_ptr<NodeObstacleContact> &contact,
                                                             const Options                              &options) {
        const ExecutionSpace es = f->execution_space();
        if (options.format == op_type::BSR) {
            // Its values are changed in place after `op` (the curvature, the shifts), which a copy
            // taken there would miss: stored in real_t or, assembled so, in float.
            if (options.storage_bytes != (int)sizeof(real_t) && options.storage_bytes != 4) {
                SFEM_ERROR("ObstacleTangent: storage_bytes %d (4 or %d)\n", options.storage_bytes, (int)sizeof(real_t));
                return nullptr;
            }
            auto ret     = std::make_shared<AssembledTangent>();
            ret->tangent = BSRTangent::create(f, es, options.storage_bytes);
            ret->contact = contact;
            if (!ret->tangent) return nullptr;
            return ret;
        }

        if (options.format != op_type::INEXACT && options.format != op_type::MATRIX_FREE) {
            SFEM_ERROR("ObstacleTangent: format \"%s\" (%s, %s or %s)\n",
                       options.format.c_str(),
                       op_type::BSR,
                       op_type::INEXACT,
                       op_type::MATRIX_FREE);
            return nullptr;
        }
        if (f->space()->mesh()->comm()->size() > 1 && es != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("ObstacleTangent: format %s runs distributed on the host only; use %s\n", options.format.c_str(), op_type::BSR);
            return nullptr;
        }
        if (options.format == op_type::INEXACT) {
            // Before the first store is built: the options decide its type.
            f->set_inexact_options(options.inexact);
            if (!f->tangent_is_symmetric()) {
                SFEM_ERROR("ObstacleTangent: the stored tangent is not symmetric, and MPRGP needs a symmetric one\n");
                return nullptr;
            }
        }

        auto ret       = std::make_shared<ComposedTangent>();
        ret->f         = f;
        ret->contact   = contact;
        ret->format    = options.format;
        ret->es        = es;
        ret->ndofs     = f->space()->n_dofs();
        ret->blas      = blas<real_t>(es);
        ret->state     = create_buffer<real_t>(ret->ndofs, es);
        ret->reference = create_buffer<real_t>(ret->ndofs, es);
        ret->tmp       = create_buffer<real_t>(ret->ndofs, es);
        ret->blocks    = create_buffer<real_t>(6 * contact->size(), es);
        ret->build();
        return ret;
    }

}  // namespace sfem
