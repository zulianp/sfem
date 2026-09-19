#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    class InertiaPotential final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        /// `es` is where the caller's vectors live.  Everything this operator
        /// does to a vector goes through `sfem::blas<real_t>(es)`, so it runs
        /// on whichever side the caller solves on.
        explicit InertiaPotential(const std::shared_ptr<FunctionSpace> &space,
                                  ExecutionSpace es = EXECUTION_SPACE_HOST);
        ~InertiaPotential() override;

        const char *name() const override { return "InertiaPotential"; }

        //! 1/2 alpha (x - u_hat)^T M (x - u_hat)

        bool energy_or_potential_based() const override { return true; }
        //! Where this operator runs, so `Function` allocates the buffers it
        //! writes on the same side.
        ExecutionSpace execution_space() const override;
        bool        is_linear() const override { return true; }

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

        /// The block-diagonal, upper-triangle-packed format.
        ///
        /// A lumped mass times `alpha` is `alpha * m * I` on each node's block,
        /// so only the packed diagonal entries take a contribution and the
        /// off-diagonals are left alone.  It exists because an energy material
        /// holding a scheme forwards every assembly to this operator, and a
        /// format that quietly skipped the inertia would assemble a tangent
        /// that does not match the residual beside it.
        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override;

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
