/**
 * @file sfem_OpFactory.hpp
 * @brief Factory for creating finite element operators
 * 
 * This file defines the Factory class, which provides a centralized registry
 * and creation mechanism for all finite element operators in the SFEM library.
 * The factory supports both regular operators and boundary operators.
 */

#pragma once

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Factory for creating finite element operators
     * 
     * The Factory class provides a centralized registry and creation mechanism
     * for all finite element operators. It uses a singleton pattern to maintain
     * a global registry of operator creation functions.
     * 
     * The factory supports:
     * - Registration of operator creation functions
     * - Automatic operator creation by name
     * - Support for both regular and boundary operators
     * - GPU operator variants (with "gpu:" prefix)
     * - Semi-structured mesh operator variants (with "ss:" prefix)
     * 
     * Usage example:
     * @code
     * auto op = Factory::create_op(space, "Laplacian");
     * auto boundary_op = Factory::create_boundary_op(space, boundary_elements, "BoundaryMass");
     * @endcode
     */
    class Factory {
    public:
        /**
         * @brief Function type for creating regular operators
         * @param space Function space for the operator
         * @return Unique pointer to the created operator
         */
        using FactoryFunction = std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &)>;
        
        /**
         * @brief Function type for creating boundary operators
         * @param space Function space for the operator
         * @param boundary_elements Boundary element connectivity
         * @return Unique pointer to the created operator
         */
        using FactoryFunctionBoundary = std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &, const std::shared_ptr<Buffer<idx_t *>> &)>;
        
        /**
         * @brief Destructor
         */
        ~Factory();
        
        /**
         * @brief Get the singleton instance of the factory
         * @return Reference to the factory instance
         */
        static Factory &instance();
        
        /**
         * @brief Register an operator creation function
         * @param name Operator name
         * @param factory_function Function to create the operator
         */
        static void register_op(const std::string &name, FactoryFunction factory_function);
        
        /**
         * @brief Create an operator by name
         * @param space Function space
         * @param name Operator name
         * @return Shared pointer to the created operator, or nullptr if not found
         * 
         * The factory automatically adds "ss:" prefix for semi-structured meshes.
         */
        static std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space, const char *name);
        
        /**
         * @brief Create a GPU operator by name
         * @param space Function space
         * @param name Operator name (without "gpu:" prefix)
         * @return Shared pointer to the created operator, or nullptr if not found
         * 
         * This method automatically adds the "gpu:" prefix to the operator name.
         */
        static std::shared_ptr<Op> create_op_gpu(const std::shared_ptr<FunctionSpace> &space, const char *name);
        
        /**
         * @brief Create a boundary operator by name
         * @param space Function space
         * @param boundary_elements Boundary element connectivity
         * @param name Operator name
         * @return Shared pointer to the created operator, or nullptr if not found
         */
        static std::shared_ptr<Op> create_boundary_op(const std::shared_ptr<FunctionSpace>   &space,
                                                      const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
                                                      const char                             *name);

    private:
        /**
         * @brief Private constructor for singleton pattern
         */
        Factory();
        
        /**
         * @brief Private registration method
         * @param name Operator name
         * @param factory_function Function to create the operator
         */
        void private_register_op(const std::string &name, FactoryFunction factory_function);

        /**
         * @brief Implementation details
         */
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /**
     * @brief Add "gpu:" prefix to operator name
     * @param name Original operator name
     * @return Operator name with "gpu:" prefix
     */
    std::string d_op_str(const std::string &name);

} // namespace sfem 