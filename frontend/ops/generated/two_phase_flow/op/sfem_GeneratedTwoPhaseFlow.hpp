#pragma once

#include "sfem_Op.hpp"
#include "sfem_NeumannConditions.hpp"

namespace sfem {
    class GeneratedTwoPhaseFlow final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space);
        ~GeneratedTwoPhaseFlow() override;

        const char *name() const override { return "GeneratedTwoPhaseFlow"; }
        bool is_linear() const override { return false; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;
        double flops_value() const override;
        double flops_gradient() const override;
        double flops_apply() const override;
        size_t memory_traffic_bytes_value() const override;
        size_t memory_traffic_bytes_gradient() const override;
        size_t memory_traffic_bytes_apply() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;
        int update(const real_t *const x) override;
        int update(const real_t *const previous, const real_t *const current) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        void set_option(const std::string &name, bool val) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
