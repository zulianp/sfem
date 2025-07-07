/**
 * @file sfem_VectorLaplacian.hpp
 * @brief Vector Laplacian operator for finite element analysis
 * 
 * This file defines the VectorLaplacian operator, which implements the discrete
 * form of the vector Laplacian operator -∇² for vector fields. This operator
 * is commonly used in fluid dynamics, electromagnetics, and other vector field problems.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Vector Laplacian operator for vector fields
     * 
     * The VectorLaplacian operator implements the discrete form of the vector Laplacian:
     * 
     * -∇²u = f
     * 
     * where:
     * - u is the vector field (velocity, electric field, etc.)
     * - f is the vector source term
     * 
     * This operator is used in various applications:
     * - Fluid dynamics (Stokes equations)
     * - Electromagnetics (vector Helmholtz equation)
     * - Elasticity (vector wave equation)
     * - Vector diffusion problems
     * 
     * The operator supports:
     * - Vector function spaces (block_size > 1)
     * - Various element types (HEX8, TET4, etc.)
     * - Optional FFF (Fast Finite Element) optimization
     * - Level-of-refinement (LOR) and derefinement
     * - Performance tracking
     */
    class VectorLaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        enum ElemType element_type { INVALID }; ///< Element type

        std::shared_ptr<Buffer<jacobian_t>> fff; ///< Fast Finite Element data (optional)

        long   calls{0};      ///< Number of apply() calls for performance tracking
        double total_time{0}; ///< Total time spent in apply() for performance tracking

        const char *name() const override { return "VectorLaplacian"; }
        inline bool is_linear() const override { return true; }

        /**
         * @brief Create a VectorLaplacian operator
         * @param space Function space (must have block_size > 1)
         * @return Unique pointer to the operator
         * 
         * The operator requires a vector function space (block_size > 1).
         * Optionally creates FFF data for performance optimization if
         * SFEM_VECTOR_LAPLACIAN_FFF environment variable is set to 1.
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
        VectorLaplacian(const std::shared_ptr<FunctionSpace> &space);

        /**
         * @brief Destructor
         * 
         * Prints performance statistics if SFEM_PRINT_THROUGHPUT is enabled.
         */
        ~VectorLaplacian();

        // Matrix assembly methods (not implemented)
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
        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
    };

} // namespace sfem 