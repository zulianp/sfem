/**
 * @file sfem_SemiStructuredLinearElasticity.hpp
 * @brief Semi-structured linear elasticity operator for finite element analysis
 *
 * This file defines the SemiStructuredLinearElasticity operator, which implements
 * the discrete linear elasticity equations optimized for semi-structured meshes.
 * Semi-structured meshes are regular Cartesian grids with local refinements,
 * allowing for efficient matrix-free implementations.
 */

#pragma once
#include "sfem_Op.hpp"

namespace sfem {

    // Semi-structured Kelvin-Voigt Newmark operator (matrix-free apply)
    class SemiStructuredKelvinVoigtNewmark : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };
        std::shared_ptr<Buffer<real_t>> vel_[3];
        std::shared_ptr<Buffer<real_t>> acc_[3];

        // KV-Newmark parameters
        real_t k{4.0}, K{3.0}, eta{0.1}, dt{0.1}, gamma{0.5}, beta{0.25}, rho{1.0};
        bool   use_affine_approximation{true};

        long   calls{0};
        double total_time{0};

        ~SemiStructuredKelvinVoigtNewmark();

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        void set_option(const std::string &name, bool val) override;
        void set_field(const char* name, const std::shared_ptr<Buffer<real_t>>& vel, int component) override;

        std::shared_ptr<Op> clone() const override;
        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        const char *name() const override;
        inline bool is_linear() const override { return true; }
        int initialize(const std::vector<std::string> &block_names = {}) override;
        SemiStructuredKelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space);

        // Not implemented matrix assembly for KV here
        int hessian_crs(const real_t *const, const count_t *const, const idx_t *const, real_t *const) override;
        int hessian_crs_sym(const real_t *const, const count_t *const, const idx_t *const, real_t *const, real_t *const) override;
        int hessian_bsr(const real_t *const, const count_t *const, const idx_t *const, real_t *const) override;
        int hessian_bcrs_sym(const real_t *const, const count_t *const, const idx_t *const, const ptrdiff_t, real_t **const, real_t **const) override;
        int hessian_block_diag_sym(const real_t *const, real_t *const) override;
        int hessian_diag(const real_t *const, real_t *const) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
    };

}  // namespace sfem