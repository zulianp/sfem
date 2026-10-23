#pragma once

#include <memory>

#include "sfem_TimeScheme.hpp"

namespace sfem {

    class FunctionSpace;

    /// A time scheme seen from a coarser level of a multigrid hierarchy, for the operator a
    /// derefined material holds: the fine scheme's shift and weights, read live, and its inertia
    /// rediscretised on the coarse space (InertiaPotential::derefine_op), its alpha the fine one's at
    /// every use. A coarse level only applies and assembles, so there is no history and no stepping:
    /// those refuse.
    std::shared_ptr<TimeScheme> derefine_time_scheme(const std::shared_ptr<TimeScheme>    &fine,
                                                     const std::shared_ptr<FunctionSpace> &coarse);

}  // namespace sfem
