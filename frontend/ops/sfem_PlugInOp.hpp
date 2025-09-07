/**
 * @file sfem_PlugInOp.hpp
 * @brief Dynamic plug-in operator loader
 */

#pragma once

#include "sfem_Op.hpp"

#include <memory>
#include <string>

namespace sfem {

    /**
     * @brief Operator that forwards to functions loaded from a shared library
     *
     * Loads element-specific kernels via dlopen/dlsym using a naming convention:
     *   <opname>_<elem>_gradient, <opname>_<elem>_apply
     * where <elem> is one of {hex8, tet4, ...} in lowercase.
     */
    class PlugInOp final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space, const std::string &opname);

        const char *name() const override { return name_.c_str(); }
        bool        is_linear() const override { return false; }

        int initialize(const std::vector<std::string> &block_names = {}) override;

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

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

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override;
        int hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) override;
        int hessian_diag(const real_t *const x, real_t *const out) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override { return SFEM_SUCCESS; }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;
        void override_element_types(const std::vector<enum ElemType> &element_types) override;

        ~PlugInOp();

    private:
        PlugInOp(const std::shared_ptr<FunctionSpace> &space, const std::string &opname);

        class Impl;
        std::unique_ptr<Impl> impl_;
        std::string           name_;
    };

}  // namespace sfem

