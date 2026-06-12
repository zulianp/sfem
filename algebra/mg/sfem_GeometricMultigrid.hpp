#ifndef SFEM_GEOMETRIC_MULTIGRID_HPP
#define SFEM_GEOMETRIC_MULTIGRID_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_defs.hpp"

#include <memory>
#include <vector>

namespace sfem {

    struct MultigridData {
        std::vector<std::shared_ptr<Function>>         functions;
        std::vector<std::shared_ptr<Operator<real_t>>> restrictions;
        std::vector<std::shared_ptr<Operator<real_t>>> prolongations;
    };

    std::shared_ptr<MultigridData> create_gmg_data(const std::shared_ptr<Function> &f);

    std::vector<std::shared_ptr<Operator<real_t>>> create_gmg_operators(const std::shared_ptr<MultigridData> &data,
                                                                        const OperatorType                    op_type);

    std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> create_gmg_default_smoothers_and_solver(
            const std::shared_ptr<MultigridData>           &data,
            std::vector<std::shared_ptr<Operator<real_t>>> &ops,
            const int                                       smoothing_steps,
            const bool                                      emable_mixed_precision);

}  // namespace sfem

#endif