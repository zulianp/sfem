/**
 * @file sfem_LinearElasticity.hpp
 * @brief Linear elasticity operator for finite element analysis
 *
 * This file defines the LinearElasticity operator, which implements
 * the discrete linear elasticity equations for solid mechanics problems.
 * The operator supports various element types and matrix formats.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Linear elasticity operator for solid mechanics
     *
     * The LinearElasticity operator implements the discrete form of the
     * linear elasticity equations:
     *
     * -∇·σ = f
     * σ = λ(∇·u)I + 2μ(∇u + ∇u^T)/2
     *
     * where:
     * - σ is the Cauchy stress tensor
     * - u is the displacement field
     * - λ and μ are the Lamé parameters
     * - f is the body force
     *
     * The operator supports:
     * - Various element types (HEX8, TET4, etc.)
     * - Multiple matrix formats (CRS, BSR, diagonal)
     * - Level-of-refinement (LOR) and derefinement
     * - Performance optimization with precomputed Jacobians
     * - Multi-domain operations via MultiDomainOp
     */
    class LinearElasticity final : public Op {
    public:
        const char *name() const override { return "LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        // Accessors for compatibility with semi-structured wrappers
        real_t get_mu() const;
        void   set_mu(real_t val);
        real_t get_lambda() const;
        void   set_lambda(real_t val);

        /**
         * @brief Create a LinearElasticity operator
         * @param space Function space
         * @return Unique pointer to the operator
         *
         * The operator reads material parameters from environment variables:
         * - SFEM_SHEAR_MODULUS: Shear modulus μ (default: 1.0)
         * - SFEM_FIRST_LAME_PARAMETER: First Lamé parameter λ (default: 1.0)
         */
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        /**
         * @brief Create a level-of-refinement (LOR) version
         * @param space Function space for LOR operator
         * @return Shared pointer to LOR operator
         */
        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;

        /**
         * @brief Create a derefined version
         * @param space Function space for derefined operator
         * @return Shared pointer to derefined operator
         */
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        /**
         * @brief Initialize the operator
         * @param block_names Optional list of block names to initialize
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * Sets up the MultiDomainOp for multi-block operations.
         * For HEX8 elements, this precomputes Jacobian determinants and adjugates
         * to optimize matrix-vector products.
         */
        int initialize(const std::vector<std::string> &block_names = {}) override;

        /**
         * @brief Constructor
         * @param space Function space
         */
        LinearElasticity(const std::shared_ptr<FunctionSpace> &space);

        /**
         * @brief Destructor
         *
         * Prints performance statistics if SFEM_PRINT_THROUGHPUT is enabled.
         */
        ~LinearElasticity();

        // Matrix assembly methods
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

        int hessian_diag(const real_t *const, real_t *const out) override;

        // Vector operations
        int                 gradient(const real_t *const x, real_t *const out) override;
        int                 apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
        int                 value(const real_t *x, real_t *const out) override;
        int                 report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;
        void override_element_types(const std::vector<enum ElemType> &element_types) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem