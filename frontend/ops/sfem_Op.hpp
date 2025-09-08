/**
 * @file sfem_Op.hpp
 * @brief Base operator class for finite element operators
 *
 * This file defines the abstract base class for all finite element operators
 * in the SFEM library. Operators represent discrete differential operators
 * that can be applied to finite element functions.
 */

#pragma once

#include "sfem_Buffer.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_defs.h"
// #include "sfem_Function.hpp"
#include "sfem_glob.hpp"

#include <vector>
#include <string>

namespace sfem {

    /**
     * @brief Abstract base class for finite element operators
     *
     * The Op class defines the interface for all finite element operators
     * in the SFEM library. Operators can represent various differential
     * operators like Laplacian, Linear Elasticity, Mass matrices, etc.
     *
     * Each operator must implement:
     * - Hessian assembly (matrix form)
     * - Gradient computation
     * - Operator application (matrix-vector product)
     * - Value computation (energy/functional evaluation)
     *
     * Operators can be linear or nonlinear, and support various matrix
     * formats (CRS, BSR, diagonal, etc.).
     */
    class Op {
    public:
        virtual ~Op() = default;

        /**
         * @brief Get the name of the operator
         * @return String identifier for the operator
         */
        virtual const char *name() const = 0;

        /**
         * @brief Check if the operator is linear
         * @return true if linear, false if nonlinear
         */
        virtual bool is_linear() const = 0;

        /**
         * @brief Initialize the operator
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method is called after construction to perform any
         * necessary setup, such as precomputing Jacobians or element matrices.
         */
        virtual int initialize(const std::vector<std::string> &block_names = {}) { return SFEM_SUCCESS; }

        /**
         * @brief Assemble the Hessian matrix in CRS format
         * @param x Current solution vector
         * @param rowptr Row pointer array for CRS format
         * @param colidx Column index array for CRS format
         * @param values Matrix values array (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method assembles the Hessian matrix in Compressed Row Storage (CRS) format.
         * The matrix represents the second derivative of the operator with respect to the solution.
         */
        virtual int hessian_crs(const real_t *const  x,
                                const count_t *const rowptr,
                                const idx_t *const   colidx,
                                real_t *const        values) = 0;

        /**
         * @brief Assemble the symmetric Hessian matrix in CRS format
         * @param x Current solution vector
         * @param rowptr Row pointer array for CRS format
         * @param colidx Column index array for CRS format
         * @param diag_values Diagonal values array (output)
         * @param off_diag_values Off-diagonal values array (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method assembles the symmetric Hessian matrix, separating diagonal and off-diagonal entries.
         */
        virtual int hessian_crs_sym(const real_t *const  x,
                                    const count_t *const rowptr,
                                    const idx_t *const   colidx,
                                    real_t *const        diag_values,
                                    real_t *const        off_diag_values) {
            SFEM_ERROR("hessian_crs_sym not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Assemble the symmetric Hessian matrix in block CRS format
         * @param x Current solution vector
         * @param rowptr Row pointer array for CRS format
         * @param colidx Column index array for CRS format
         * @param block_stride Block size (ptrdiff_t)
         * @param diag_values Diagonal values array (output, real_t**)
         * @param off_diag_values Off-diagonal values array (output, real_t**)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method assembles the symmetric Hessian matrix in block CRS format.
         */
        virtual int hessian_bcrs_sym(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     const ptrdiff_t      block_stride,
                                     real_t **const       diag_values,
                                     real_t **const       off_diag_values) {
            SFEM_ERROR("hessian_bcrs_sym not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Assemble the block diagonal symmetric Hessian matrix
         * @param x Current solution vector
         * @param values Block diagonal values array (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method assembles only the block diagonal entries of the symmetric Hessian matrix.
         */
        virtual int hessian_block_diag_sym(const real_t *const x, real_t *const values) {
            SFEM_ERROR("hessian_block_diag_sym not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Assemble the block diagonal symmetric Hessian matrix in structure-of-arrays format
         * @param x Current solution vector
         * @param values Array of block diagonal values arrays (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         *
         * This method assembles the block diagonal entries in structure-of-arrays format.
         */
        virtual int hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) {
            SFEM_ERROR("hessian_block_diag_sym_soa not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Assemble the Hessian matrix in BSR format
         * @param x Current solution vector
         * @param rowptr Row pointer array for BSR format
         * @param colidx Column index array for BSR format
         * @param values Matrix values array (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int hessian_bsr(const real_t *const /*x*/,
                                const count_t *const /*rowptr*/,
                                const idx_t *const /*colidx*/,
                                real_t *const /*values*/) {
            SFEM_ERROR("BSR assembly not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Assemble the diagonal of the Hessian matrix
         * @param x Current solution vector
         * @param values Diagonal values (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) {
            SFEM_ERROR("Diagonal assembly not implemented for this operator");
            return SFEM_FAILURE;
        }

        /**
         * @brief Compute the gradient of the operator
         * @param x Current solution vector
         * @param out Gradient vector (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int gradient(const real_t *const x, real_t *const out) = 0;

        /**
         * @brief Apply the operator (matrix-vector product)
         * @param x Current solution vector (may be nullptr for linear operators)
         * @param h Input vector
         * @param out Output vector
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int apply(const real_t *const x, const real_t *const h, real_t *const out) = 0;

        /**
         * @brief Compute the value/energy of the operator
         * @param x Current solution vector
         * @param out Energy value (output)
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int value(const real_t *x, real_t *const out) = 0;

        /**
         * @brief Report operator statistics or debug information
         * @param x Current solution vector
         * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
         */
        virtual int report(const real_t *const /*x*/) { return SFEM_SUCCESS; }

        /**
         * @brief Set operator-specific options
         * @param name Option name
         * @param val Option value
         */
        virtual void set_option(const std::string & /*name*/, bool /*val*/) {}

        virtual void set_value_in_block(const std::string & /*block_name*/,
                                        const std::string & /*var_name*/,
                                        const real_t /*value*/) {}

        /**
         * @brief Create a clone of this operator
         * @return Shared pointer to the cloned operator, or nullptr if not supported
         *
         * This method allows operators to be cloned for use in different contexts.
         * The default implementation returns nullptr, indicating that cloning is not supported.
         * Derived classes should override this method if they support cloning.
         */
        virtual std::shared_ptr<Op> clone() const { return nullptr; }

        /**
         * @brief Create a level-of-refinement (LOR) version of this operator
         * @param space Function space for the LOR operator
         * @return Shared pointer to the LOR operator, or nullptr if not supported
         */
        virtual std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) { return nullptr; }

        /**
         * @brief Create a derefined version of this operator
         * @param space Function space for the derefined operator
         * @return Shared pointer to the derefined operator, or nullptr if not supported
         */
        virtual std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) { return nullptr; }

        /**
         * @brief Get the execution space of this operator
         * @return Execution space (CPU, GPU, etc.)
         */
        virtual ExecutionSpace execution_space() const { return EXECUTION_SPACE_HOST; }

        /**
         * @brief Set a field for operators that depend on external fields
         * @param name Field name
         * @param v Field values
         * @param component Field component
         */
        virtual void set_field(const char * /*name*/, const std::shared_ptr<Buffer<real_t>> & /*v*/, const int /*component*/) {}

        virtual bool is_no_op() const { return false; }

        virtual void override_element_types(const std::vector<enum ElemType> &element_types) {}
    };

    /**
     * @brief No-operation operator
     *
     * A trivial operator that does nothing. Useful as a placeholder
     * or for testing purposes.
     */
    class NoOp final : public Op {
    public:
        const char *name() const override { return "NoOp"; }
        bool        is_linear() const override { return true; }
        int         hessian_crs(const real_t *const /*x*/,
                                const count_t *const /*rowptr*/,
                                const idx_t *const /*colidx*/,
                                real_t *const /*values*/) override {
            return SFEM_SUCCESS;
        }
        int gradient(const real_t *const /*x*/, real_t *const /*out*/) override { return SFEM_SUCCESS; }
        int apply(const real_t *const /*x*/, const real_t *const /*h*/, real_t *const /*out*/) override { return SFEM_SUCCESS; }
        int value(const real_t * /*x*/, real_t *const /*out*/) override { return SFEM_SUCCESS; }
        std::shared_ptr<Op> clone() const override { return std::make_shared<NoOp>(); }
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override { return std::make_shared<NoOp>(); }
        bool is_no_op() const override { return true; }
    };

    /**
     * @brief Create a no-operation operator
     * @return Shared pointer to a NoOp instance
     */
    std::shared_ptr<Op> no_op();

}  // namespace sfem