/**
 * @file sfem_BoundaryMass.hpp
 * @brief Boundary mass matrix operator for finite element analysis
 *
 * This file defines the BoundaryMass operator, which implements the discrete
 * mass matrix for boundary elements. This operator is commonly used for
 * boundary conditions, surface integrals, and boundary-coupled problems.
 */

#pragma once
#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Boundary mass matrix operator for boundary elements
     *
     * The BoundaryMass operator implements the discrete mass matrix for boundary elements:
     *
     * M_ij = ∫_∂Ω φ_i φ_j dS
     *
     * where:
     * - φ_i, φ_j are finite element basis functions
     * - ∂Ω is the boundary of the computational domain
     *
     * This operator is used in various applications:
     * - Boundary conditions (Robin, impedance)
     * - Surface integrals and boundary-coupled problems
     * - Boundary element methods
     * - Interface problems
     * - Boundary stabilization
     *
     * The operator supports:
     * - Various boundary element types (QUAD4, TRI3, etc.)
     * - Both scalar and vector function spaces
     * - CRS matrix format for assembly
     * - Matrix-vector products for application
     * - Automatic boundary element detection
     */
    class BoundaryMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace>   space;                     ///< Function space for the operator
        std::shared_ptr<Buffer<idx_t *>> boundary_elements;         ///< Boundary element connectivity
        enum ElemType                    element_type { INVALID };  ///< Element type
        const char                      *name() const override { return "BoundaryMass"; }
        inline bool                      is_linear() const override { return true; }

        /**
         * @brief Create a BoundaryMass operator
         * @param space Function space
         * @param boundary_elements Boundary element connectivity
         * @return Unique pointer to the operator
         *
         * The operator requires boundary element connectivity to identify
         * which elements are on the boundary.
         */
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace>   &space,
                                          const std::shared_ptr<Buffer<idx_t *>> &boundary_elements);

        /**
         * @brief Initialize the operator
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * Validates boundary element connectivity and determines element type.
         */
        int initialize(const std::vector<std::string> &block_names = {}) override;

        /**
         * @brief Constructor
         * @param space Function space
         */
        BoundaryMass(const std::shared_ptr<FunctionSpace> &space);

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
    };
}  // namespace sfem