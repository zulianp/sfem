/**
 * @file sfem_Laplacian.hpp
 * @brief Laplacian operator for finite element analysis
 * 
 * This file defines the Laplacian operator, which implements the discrete
 * form of the Laplace operator -∇² for scalar fields. This operator is
 * commonly used in heat conduction, electrostatics, and other diffusion problems.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Laplacian operator for scalar fields
     * 
     * The Laplacian operator implements the discrete form of the Laplace operator:
     * 
     * -∇²u = f
     * 
     * where:
     * - u is the scalar field (temperature, potential, etc.)
     * - f is the source term
     * 
     * This operator is used in various applications:
     * - Heat conduction problems
     * - Electrostatic problems
     * - Diffusion equations
     * - Poisson equations
     * 
     * The operator supports:
     * - Various element types (HEX8, TET4, etc.)
     * - Multiple matrix formats (CRS, diagonal)
     * - Level-of-refinement (LOR) and derefinement
     * - Performance tracking
     */
    class Laplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        enum ElemType element_type { INVALID }; ///< Element type

        long   calls{0};      ///< Number of apply() calls for performance tracking
        double total_time{0}; ///< Total time spent in apply() for performance tracking

        const char *name() const override { return "Laplacian"; }
        inline bool is_linear() const override { return true; }

        /**
         * @brief Create a Laplacian operator
         * @param space Function space (must have block_size == 1)
         * @return Unique pointer to the operator
         * 
         * The operator requires a scalar function space (block_size == 1).
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
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         * 
         * Currently a no-op, but may be extended for future optimizations.
         */
        int initialize() override { return SFEM_SUCCESS; }

        /**
         * @brief Constructor
         * @param space Function space
         */
        Laplacian(const std::shared_ptr<FunctionSpace> &space);

        /**
         * @brief Destructor
         * 
         * Prints performance statistics if SFEM_PRINT_THROUGHPUT is enabled.
         */
        ~Laplacian();

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

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override;

        // Vector operations
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
    };

} // namespace sfem 