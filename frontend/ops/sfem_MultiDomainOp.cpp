#include "sfem_MultiDomainOp.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_logger.h"

namespace sfem {

    MultiDomainOp::MultiDomainOp(const std::shared_ptr<FunctionSpace> &space, const std::vector<std::string> &block_names) {
        if (block_names.empty()) {
            for (auto &block : space->mesh_ptr()->blocks()) {
                domains_[block->name()] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
            }
        } else {
            for (auto &block_name : block_names) {
                auto block = space->mesh_ptr()->find_block(block_name);
                if (!block) {
                    SFEM_ERROR("Block %s not found", block_name.c_str());
                }
                domains_[block_name] = OpDomain{block->element_type(), block, std::make_shared<Parameters>()};
            }
        }
    }

    int MultiDomainOp::iterate(const std::function<int(const OpDomain &)> &func) {
        for (auto &domain : domains_) {
            int err = func(domain.second);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }
        return SFEM_SUCCESS;
    }

    void MultiDomainOp::override_element_types(const std::vector<enum ElemType> &element_types) {
        size_t i = 0;
        for (auto &domain : domains_) {
            assert(i < element_types.size());
            domain.second.element_type = element_types[i];
            i++;
        }
    }

    std::shared_ptr<MultiDomainOp> MultiDomainOp::lor_op(const std::shared_ptr<FunctionSpace> &space,
                                                          const std::vector<std::string> &block_names) {
        auto ret = std::make_shared<MultiDomainOp>(space, block_names);

        for (auto &domain : ret->domains_) {
            domain.second.element_type = macro_type_variant(domain.second.element_type);
        }

        return ret;
    }

    std::shared_ptr<MultiDomainOp> MultiDomainOp::derefine_op(const std::shared_ptr<FunctionSpace> &space,
                                                               const std::vector<std::string> &block_names) {
        auto ret = std::make_shared<MultiDomainOp>(space, block_names);

        for (auto &domain : ret->domains_) {
            domain.second.element_type = macro_base_elem(domain.second.element_type);
        }

        return ret;
    }

    void MultiDomainOp::print_info() {
        for (auto &domain : domains_) {
            printf("Domain %s: %s\n", domain.first.c_str(), domain.second.block->name().c_str());
        }
    }

} // namespace sfem 
