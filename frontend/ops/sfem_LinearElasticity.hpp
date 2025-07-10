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
     */
    class LinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        enum ElemType element_type { INVALID }; ///< Element type

        real_t mu{1};     ///< Shear modulus (second Lamé parameter)
        real_t lambda{1}; ///< First Lamé parameter

        long   calls{0};      ///< Number of apply() calls for performance tracking
        double total_time{0}; ///< Total time spent in apply() for performance tracking

        /**
         * @brief Jacobian storage for performance optimization
         * 
         * Precomputed Jacobian determinants and adjugates for HEX8 elements
         * to avoid repeated computation during matrix-vector products.
         */
        class Jacobians {
        public:
            std::shared_ptr<Buffer<jacobian_t>> adjugate;   ///< Adjugate matrices
            std::shared_ptr<Buffer<jacobian_t>> determinant; ///< Determinants

            /**
             * @brief Constructor
             * @param n_elements Number of elements
             * @param size_adjugate Size of adjugate storage per element
             */
            Jacobians(const ptrdiff_t n_elements, const int size_adjugate)
                : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
                  determinant(sfem::create_host_buffer<jacobian_t>(n_elements)) {}
        };

        std::shared_ptr<Jacobians> jacobians; ///< Precomputed Jacobians

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

        const char *name() const override { return "LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        /**
         * @brief Initialize the operator
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         * 
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
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
    };

} // namespace sfem 