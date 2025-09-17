#pragma once
#include "sfem_Op.hpp"

namespace sfem {
    class NeoHookeanOgden final : public Op {
    public:
        static std::unique_ptr<Op>     create(const std::shared_ptr<FunctionSpace> &space);
        std::shared_ptr<Op>            lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op>            derefine_op(const std::shared_ptr<FunctionSpace> &space) override;
        const char                    *name() const override { return "NeoHookeanOgden"; }
        inline bool                    is_linear() const override { return true; }
        int                            initialize(const std::vector<std::string> &block_names = {}) override;
        NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space);
        int                 hessian_crs(const real_t *const  x,
                                        const count_t *const rowptr,
                                        const idx_t *const   colidx,
                                        real_t *const        values) override;
        int                 hessian_diag(const real_t *const x, real_t *const out) override;
        int                 gradient(const real_t *const x, real_t *const out) override;
        int                 apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int                 value(const real_t *x, real_t *const out) override;
        int                 report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
        ~NeoHookeanOgden() override;

        int update(const real_t *const x) override;
        int value_steps(const real_t *x, const real_t *h, const int nsteps, const real_t *const steps, real_t *const out) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem