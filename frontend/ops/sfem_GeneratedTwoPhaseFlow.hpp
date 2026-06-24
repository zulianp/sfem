#pragma once

#include "sfem_Op.hpp"
#include "sfem_Operator.hpp"

namespace sfem {
    struct GeneratedTwoPhaseFlowStats {
        double residual_seconds{0};
        double jacobian_seconds{0};
        ptrdiff_t residual_calls{0};
        ptrdiff_t jacobian_calls{0};
    };

    class GeneratedTwoPhaseFlow final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space);
        ~GeneratedTwoPhaseFlow() override;

        const char *name() const override { return "GeneratedTwoPhaseFlow"; }
        bool is_linear() const override { return false; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;
        int update(const real_t *const x) override;
        int update(const real_t *const x_prev, const real_t *const x_curr) override;
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;

        int hessian_crs(const real_t *const,
                        const count_t *const,
                        const idx_t *const,
                        real_t *const) override;
        int value(const real_t *, real_t *const) override;

        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;

        std::shared_ptr<Operator<real_t>> make_linear_operator(
                const real_t *current,
                std::function<void(const real_t *const, real_t *const)> apply_constraints = {});
        void reset_stats();
        GeneratedTwoPhaseFlowStats stats() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
