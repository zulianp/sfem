#ifndef SFEM_GEOMETRIC_MULTIGRID_HPP
#define SFEM_GEOMETRIC_MULTIGRID_HPP

#include "sfem_ForwardDeclarations.hpp"

#include <memory>
#include <vector>

namespace sfem {

    template <typename T>
    struct MultigridHierarchy {
        std::vector<std::shared_ptr<Operator<T>>> op;
        std::vector<std::shared_ptr<Operator<T>>> restriction;
        std::vector<std::shared_ptr<Operator<T>>> prolongation;
    };

    std::shared_ptr<MultigridHierarchy<real_t>> create_gmg_hierarchy(const std::shared_ptr<Function> &f);

}  // namespace sfem

#endif