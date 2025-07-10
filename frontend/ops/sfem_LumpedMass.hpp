/**
 * @file sfem_LumpedMass.hpp
 * @brief Lumped mass matrix operator for finite element analysis
 * 
 * This file defines the LumpedMass operator, which implements the discrete
 * lumped mass matrix for finite element discretizations. The lumped mass matrix
 * is a diagonal approximation of the full mass matrix, commonly used for
 * explicit time integration schemes.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Lumped mass matrix operator for finite element discretizations
     * 
     * The LumpedMass operator implements the discrete lumped mass matrix:
     * 
     * M_lumped_ii = ∫_Ω φ_i dΩ
     * 
     * where:
     * - φ_i are finite element basis functions
     * - Ω is the computational domain
     * 
     * The lumped mass matrix is a diagonal approximation of the full mass matrix,
     * obtained by summing the rows of the mass matrix to the diagonal:
     * 
     * M_lumped_ii = Σ_j M_ij
     * 
     * This operator is used in various applications:
     * - Explicit time integration schemes
     * - Fast matrix-vector products (diagonal matrix)
     * - Stabilization in certain numerical schemes
     * - Preconditioning for iterative solvers
     * 
     * The operator supports:
     * - Various element types (HEX8, TET4, etc.)
     * - Both scalar and vector function spaces
     * - Diagonal matrix format only
     * - Performance optimization through diagonal structure
     */
    class LumpedMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        enum ElemType element_type { INVALID }; ///< Element type

        const char *name() const override { return "LumpedMass"; }
        inline bool is_linear() const override { return true; }

        /**
         * @brief Create a LumpedMass operator
         * @param space Function space
         * @return Unique pointer to the operator
         * 
         * The operator supports both scalar and vector function spaces.
         */
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        /**
         * @brief Initialize the operator
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         * 
         * Currently a no-op, but may be extended for future optimizations.
         */
        int initialize(const std::vector<std::string> &block_names = {}) override { return SFEM_SUCCESS; }

        /**
         * @brief Constructor
         * @param space Function space
         */
        LumpedMass(const std::shared_ptr<FunctionSpace> &space);

        // Matrix assembly methods
        int hessian_diag(const real_t *const /*x*/, real_t *const values) override;

        // Vector operations (not implemented - diagonal matrix only)
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;   
        std::shared_ptr<Op> clone() const override;
    };

} // namespace sfem 