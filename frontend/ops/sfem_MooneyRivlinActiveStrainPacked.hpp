#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    class MooneyRivlinActiveStrainPacked final : public Op {
    public:
        const char *name() const override { return "MooneyRivlinActiveStrainPacked"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        int initialize(const std::vector<std::string> &block_names = {}) override;

        MooneyRivlinActiveStrainPacked(const std::shared_ptr<FunctionSpace> &space);
        ~MooneyRivlinActiveStrainPacked();

        int update(const real_t *const u) override;

        // Matrix assembly methods
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_crs_sym(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override;

        int hessian_diag(const real_t *const x, real_t *const values) override;

        // Vector operations
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int value_steps(const real_t       *x,
                        const real_t       *h,
                        const int           nsteps,
                        const real_t *const steps,
                        real_t *const       out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;
        void override_element_types(const std::vector<enum ElemType> &element_types) override;

        // Set external fields (e.g., active strain)
        void set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &v, const int component) override;

        void set_mu(const real_t mu);
        void set_lambda(const real_t lambda);
        void set_lmda(const real_t lmda);

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bcrs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             const ptrdiff_t      block_stride,
                             real_t **const       diag_values,
                             real_t **const       off_diag_values) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem





