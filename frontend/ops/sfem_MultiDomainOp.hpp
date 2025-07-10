/**
 * @file sfem_MultiDomainOp.hpp
 * @brief Multi-domain operator support for finite element analysis
 * 
 * This file defines the OpDomain and MultiDomainOp classes that provide
 * support for operators that work across multiple mesh domains/blocks.
 */

#pragma once

#include "sfem_Op.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Mesh.hpp"

#include <map>
#include <string>
#include <functional>

namespace sfem {

    /**
     * @brief Represents a single domain within a multi-domain operator
     * 
     * Each domain contains:
     * - Element type for the domain
     * - Reference to the mesh block
     * - Parameters specific to this domain
     */
    struct OpDomain {
    public:
        enum ElemType    element_type;  ///< Element type for this domain
        SharedBlock      block;         ///< Reference to the mesh block
        SharedParameters parameters;     ///< Domain-specific parameters
        std::shared_ptr<void> user_data; ///< User data for the domain
    };

    /**
     * @brief Manages multiple domains for operators that work across multiple mesh blocks
     * 
     * This class provides a unified interface for operators that need to work
     * with multiple mesh domains/blocks. It handles:
     * - Domain initialization from block names
     * - Iteration over domains
     * - Element type overrides
     * - LOR and derefinement operations
     */
    class MultiDomainOp {
    public:
        /**
         * @brief Constructor
         * @param space Function space
         * @param block_names Optional list of specific block names to include
         * 
         * If block_names is empty, includes all blocks from the mesh.
         * Otherwise, only includes the specified blocks.
         */
        MultiDomainOp(const std::shared_ptr<FunctionSpace> &space, const std::vector<std::string> &block_names);

        /**
         * @brief Iterate over all domains with a function
         * @param func Function to apply to each domain
         * @return SFEM_SUCCESS if all iterations succeed, first error code otherwise
         * 
         * The function should take a const OpDomain& parameter and return int.
         */
        int iterate(const std::function<int(const OpDomain &)> &func);

        /**
         * @brief Override element types for all domains
         * @param element_types Vector of element types (must match domain count)
         */
        void override_element_types(const std::vector<enum ElemType> &element_types);

        /**
         * @brief Create a low-order-refinement (LOR) version
         * @param space Function space for LOR operator
         * @param block_names Block names for the new operator
         * @return Shared pointer to LOR operator
         */
        std::shared_ptr<MultiDomainOp> lor_op(const std::shared_ptr<FunctionSpace> &space,
                                              const std::vector<std::string> &block_names);

        /**
         * @brief Create a derefined version
         * @param space Function space for derefined operator
         * @param block_names Block names for the new operator
         * @return Shared pointer to derefined operator
         */
        std::shared_ptr<MultiDomainOp> derefine_op(const std::shared_ptr<FunctionSpace> &space,
                                                   const std::vector<std::string> &block_names);

        /**
         * @brief Print information about all domains
         */
        void print_info();

        /**
         * @brief Get the domains map
         * @return Reference to the domains map
         */
        const std::map<std::string, OpDomain>& domains() const { return domains_; }
         std::map<std::string, OpDomain>& domains()  { return domains_; }

        /**
         * @brief Set a value in a block
         * @param block_name Name of the block
         * @param var_name Name of the variable
         * @param value Value to set
         */
        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value);

    private:
        std::map<std::string, OpDomain> domains_;  ///< Map of domain name to domain info
    };

} // namespace sfem 
