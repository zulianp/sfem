/**
 * @file sfem_VectorLaplacian.hpp
 * @brief Vector Laplacian operator for finite element analysis
 *
 * Discrete vector Laplacian −∇² for vector fields (block_size > 1), with
 * MultiDomainOp per mesh block like sfem_Laplacian.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    class VectorLaplacian final : public Op {
    public:
        const char *name() const override { return "VectorLaplacian"; }
        inline bool is_linear() const override { return true; }
        inline ptrdiff_t n_dofs_domain() const override;
        inline ptrdiff_t n_dofs_image() const override;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        int initialize(const std::vector<std::string> &block_names = {}) override;

        VectorLaplacian(const std::shared_ptr<FunctionSpace> &space);
        ~VectorLaplacian();

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_crs_sym(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override;

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;
        void override_element_types(const std::vector<smesh::ElemType> &element_types) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
