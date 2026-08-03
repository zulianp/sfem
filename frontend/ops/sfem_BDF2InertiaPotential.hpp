#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    class BDF2InertiaPotential final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit BDF2InertiaPotential(const std::shared_ptr<FunctionSpace> &space);
        ~BDF2InertiaPotential() override;

        const char *name() const override { return "BDF2InertiaPotential"; }
        bool        is_linear() const override { return false; }

        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;
        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;
        int hessian_diag(const real_t *const x, real_t *const values) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int value_steps(const real_t       *x,
                        const real_t       *h,
                        const int           nsteps,
                        const real_t *const steps,
                        real_t *const       out) override;

        std::shared_ptr<Op> clone() const override;

        void set_alpha(real_t alpha);
        void set_density(real_t density);
        void set_u_hat(const std::shared_ptr<Buffer<real_t>> &u_hat);
        void set_mass(const std::shared_ptr<Buffer<real_t>> &mass);
        void set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &values, const int component) override;
        void set_value_in_block(const std::string &block_name, const std::string &var_name, real_t value) override;

        std::shared_ptr<Buffer<real_t>> mass() const;
        std::shared_ptr<Buffer<real_t>> u_hat() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
