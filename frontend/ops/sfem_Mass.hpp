/**
 * @file sfem_Mass.hpp
 * @brief Mass matrix operator for finite element analysis
 * 
 * This file defines the Mass operator, which implements the discrete
 * mass matrix for finite element discretizations. The mass matrix is
 * commonly used in time-dependent problems and eigenvalue computations.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Mass matrix operator for finite element discretizations
     * 
     * The Mass operator implements the discrete mass matrix:
     * 
     * M_ij = ∫_Ω φ_i φ_j dΩ
     * 
     * where:
     * - φ_i, φ_j are finite element basis functions
     * - Ω is the computational domain
     * 
     * This operator is used in various applications:
     * - Time-dependent problems (heat equation, wave equation)
     * - Eigenvalue problems
     * - L² projections
     * - Stabilization terms
     * 
     * The operator supports:
     * - Various element types (HEX8, TET4, etc.)
     * - CRS matrix format for assembly
     * - Matrix-vector products for application
     * - Scalar function spaces only (block_size == 1)
     */
    class Mass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        enum ElemType element_type { INVALID }; ///< Element type

        const char *name() const override { return "Mass"; }
        inline bool is_linear() const override { return true; }

        /**
         * @brief Create a Mass operator
         * @param space Function space (must have block_size == 1)
         * @return Unique pointer to the operator
         * 
         * The operator requires a scalar function space (block_size == 1).
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
        Mass(const std::shared_ptr<FunctionSpace> &space);

        // Matrix assembly methods
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        // Vector operations
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
    };

} // namespace sfem 